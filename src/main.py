# src/main.py
from src.pipelines.assign_engine import run_assignment


def main() -> None:
    result = run_assignment(create_products_from_excel=False)

    print("\n===== 요약 =====")
    print(f"총 배정 건수: {result.get('total_assigned', 0)}")
    print("기사별 배정 현황:")
    for rec in result.get("drivers", []):
        print(
            f" - {rec['driverId']} / {rec['name']} / career={rec['career']} / "
            f"추천={rec['desired_capacity']} / 배정={rec.get('assigned_count', 0)}"
        )


if __name__ == "__main__":
    main()
