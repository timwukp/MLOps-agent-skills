#!/usr/bin/env python3
"""Streaming data ingestion from Kafka for ML pipelines.

Consumes messages from Kafka, accumulates into micro-batches,
validates, and writes to Parquet files with checkpointing.

Usage:
    python ingest_streaming.py --broker localhost:9092 --topic features --output data/streaming/
    python ingest_streaming.py --broker localhost:9092 --topic features --output data/streaming/ --batch-size 1000
"""
import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    logger.info("Shutdown signal received, finishing current batch...")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


class StreamingIngester:
    def __init__(self, broker, topic, group_id, output_dir,
                 batch_size=1000, flush_interval_sec=60,
                 dlq_dir=None):
        self.broker = broker
        self.topic = topic
        self.group_id = group_id
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.flush_interval = flush_interval_sec
        self.dlq_dir = Path(dlq_dir) if dlq_dir else self.output_dir / "dlq"
        self.batch = []
        self.last_flush = time.time()
        self.stats = {"consumed": 0, "written": 0, "errors": 0, "dlq": 0}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dlq_dir.mkdir(parents=True, exist_ok=True)

    def create_consumer(self):
        """Create Kafka consumer."""
        try:
            from confluent_kafka import Consumer
        except ImportError:
            logger.error("confluent-kafka not installed. Install with: pip install confluent-kafka")
            sys.exit(1)

        conf = {
            "bootstrap.servers": self.broker,
            "group.id": self.group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300000,
        }
        consumer = Consumer(conf)
        consumer.subscribe([self.topic])
        logger.info(f"Connected to {self.broker}, subscribed to {self.topic}")
        return consumer

    def deserialize_message(self, msg):
        """Deserialize a Kafka message."""
        try:
            value = msg.value()
            if value is None:
                return None
            record = json.loads(value.decode("utf-8"))
            record["_kafka_offset"] = msg.offset()
            record["_kafka_partition"] = msg.partition()
            record["_kafka_timestamp"] = msg.timestamp()[1]
            return record
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Deserialization error at offset {msg.offset()}: {e}")
            self.send_to_dlq(msg.value(), str(e))
            return None

    def send_to_dlq(self, raw_message, error_reason):
        """Send failed message to dead letter queue."""
        dlq_file = self.dlq_dir / f"dlq_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        record = {
            "raw": raw_message.decode("utf-8", errors="replace") if raw_message else "",
            "error": error_reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with open(dlq_file, "a") as f:
            f.write(json.dumps(record) + "\n")
        self.stats["dlq"] += 1

    def flush_batch(self):
        """Write accumulated batch to Parquet."""
        if not self.batch:
            return

        import pandas as pd

        df = pd.DataFrame(self.batch)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.output_dir / f"batch_{timestamp}.parquet"
        df.to_parquet(output_path, compression="snappy", index=False)

        self.stats["written"] += len(self.batch)
        logger.info(f"Flushed {len(self.batch)} records to {output_path}")
        self.batch = []
        self.last_flush = time.time()

    def should_flush(self):
        """Check if we should flush the current batch."""
        if len(self.batch) >= self.batch_size:
            return True
        if time.time() - self.last_flush >= self.flush_interval:
            return True
        return False

    def run(self):
        """Main consumption loop."""
        consumer = self.create_consumer()
        logger.info(f"Starting streaming ingestion (batch_size={self.batch_size}, "
                    f"flush_interval={self.flush_interval}s)")

        try:
            while RUNNING:
                msg = consumer.poll(timeout=1.0)

                if msg is None:
                    if self.should_flush():
                        self.flush_batch()
                    continue

                if msg.error():
                    from confluent_kafka import KafkaError
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error(f"Kafka error: {msg.error()}")
                    self.stats["errors"] += 1
                    continue

                record = self.deserialize_message(msg)
                if record is not None:
                    self.batch.append(record)
                    self.stats["consumed"] += 1

                if self.should_flush():
                    self.flush_batch()
                    consumer.commit()

        except Exception as e:
            logger.error(f"Fatal error: {e}")
            raise
        finally:
            # Flush remaining records
            if self.batch:
                self.flush_batch()
                consumer.commit()
            consumer.close()
            logger.info(f"Shutdown complete. Stats: {json.dumps(self.stats)}")


def main():
    parser = argparse.ArgumentParser(description="Streaming ingestion from Kafka")
    parser.add_argument("--broker", required=True, help="Kafka broker address")
    parser.add_argument("--topic", required=True, help="Kafka topic to consume")
    parser.add_argument("--group-id", default="ml-ingestion", help="Consumer group ID")
    parser.add_argument("--output", required=True, help="Output directory for Parquet files")
    parser.add_argument("--batch-size", type=int, default=1000, help="Records per batch")
    parser.add_argument("--flush-interval", type=int, default=60, help="Max seconds between flushes")
    parser.add_argument("--dlq-dir", default=None, help="Dead letter queue directory")

    args = parser.parse_args()

    ingester = StreamingIngester(
        broker=args.broker,
        topic=args.topic,
        group_id=args.group_id,
        output_dir=args.output,
        batch_size=args.batch_size,
        flush_interval_sec=args.flush_interval,
        dlq_dir=args.dlq_dir,
    )
    ingester.run()


if __name__ == "__main__":
    main()
