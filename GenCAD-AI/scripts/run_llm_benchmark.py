from pathlib import Path

from app.config import LLMConfig
from app.domain.intent import ParsedEngineeringIntent
from app.evaluation.metrics import score_case, summarize
from app.evaluation.report import write_reports
from app.pipeline.builder import build_engineering_spec
from app.pipeline.validation import validate_spec
from app.providers.factory import build_parser
from benchmarks import BENCHMARK


def main():
    config = LLMConfig()
    parser = build_parser(config)

    benchmark = BENCHMARK
    case_metrics = []

    for case in benchmark:
        parsed: ParsedEngineeringIntent = parser.parse(case["prompt"])
        spec = build_engineering_spec(case["prompt"], parsed)
        report = validate_spec(spec)

        case_metrics.append(
            score_case(
                case=case,
                actual_intent=parsed,
                spec=spec,
                report=report,
            )
        )

    summary = summarize(config.model, case_metrics)
    json_path, md_path = write_reports(
        Path("reports"),
        summary,
        case_metrics,
    )

    print(summary.model_dump_json(indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")

    if not summary.release_gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
