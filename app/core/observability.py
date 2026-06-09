from opentelemetry import metrics, trace
from opentelemetry.instrumentation.celery import CeleryInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import start_http_server


def setup_observability(app):
    trace.set_tracer_provider(TracerProvider())

    reader = PrometheusMetricReader()
    metrics.set_meter_provider(MeterProvider(metric_readers=[reader]))
    start_http_server(8001)

    FastAPIInstrumentor.instrument_app(app)
    CeleryInstrumentor().instrument()
    HTTPXClientInstrumentor().instrument()


meter = metrics.get_meter("linguabridge")
fraud_alerts_counter = meter.create_counter(
    name="linguabridge_fraud_alerts_total",
    description="Total number of fraud alerts triggered",
)
translation_latency_histogram = meter.create_histogram(
    name="linguabridge_translation_latency_seconds",
    description="Latency of translation pipeline",
)
