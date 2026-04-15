//! `sim_run` — Policy simulator runner.
//!
//! 시나리오 YAML + Lua policy 스크립트를 받아 시뮬레이션을 실행하고
//! Session Summary를 출력한다.
//!
//! 사용 예:
//! ```
//! sim_run --scenario tests/fixtures/sim/baseline.yaml \
//!         --lua manager/scripts/policy_example.lua \
//!         --duration 30
//! ```

use clap::Parser;

#[derive(Parser)]
#[command(name = "sim_run", about = "Policy simulator runner")]
struct Args {
    /// 시나리오 YAML 파일 경로
    #[arg(long)]
    scenario: std::path::PathBuf,

    /// Lua policy 스크립트 경로
    #[arg(long)]
    lua: std::path::PathBuf,

    /// 시뮬레이션 실행 시간 (초, 기본 30)
    #[arg(long, default_value = "30")]
    duration: f64,

    /// 전체 타임라인 출력 (compact 형식)
    #[arg(long)]
    verbose: bool,

    /// trajectory를 JSON으로 저장할 경로
    #[arg(long)]
    output_json: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    env_logger::init();

    let cfg = llm_manager::sim::config::load_scenario(&args.scenario)
        .map_err(|e| anyhow::anyhow!("시나리오 로드 실패: {e}"))?;

    let adaptation = llm_manager::config::AdaptationConfig::default();

    let mut sim = llm_manager::sim::harness::Simulator::with_lua_policy(cfg, &args.lua, adaptation)
        .map_err(|e| anyhow::anyhow!("Simulator 초기화 실패: {e}"))?;

    sim.run_for(std::time::Duration::from_secs_f64(args.duration))?;

    let traj = sim.trajectory();

    if args.verbose {
        println!("{}", traj.format_timeline_compact());
        println!();
    }

    println!("{}", traj.format_session_summary());

    // Relief Table: initial vs learned
    if let (Some(initial), Some(current)) = (
        sim.policy.initial_relief_snapshot(),
        sim.policy.relief_snapshot(),
    ) {
        // format_session_summary 이미 trajectory 기반 테이블 출력
        // initial_relief_snapshot은 configured 기본값까지 포함하여 보완
        let _ = (initial, current); // 현재는 trajectory 기반으로 충분
    }

    if let Some(out) = args.output_json {
        traj.dump_json(&out)?;
        eprintln!("trajectory 저장: {}", out.display());
    }

    Ok(())
}
