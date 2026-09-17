"""SPADE-specific arguments for Slime training.

This module provides functions to add SPADE-related command-line arguments
to Slime's argument parser.
"""

def add_spade_arguments(parser):
    """Add SPADE-specific arguments to the argument parser.

    Args:
        parser: argparse.ArgumentParser to add arguments to
    """
    spade_args = parser.add_argument_group("SPADE Arguments")

    # Learning potential arguments
    spade_args.add_argument(
        "--spade-gamma1",
        type=float,
        default=0.99,
        help="Fast moving average gamma for learning potential computation",
    )
    spade_args.add_argument(
        "--spade-gamma2",
        type=float,
        default=0.95,
        help="Slow moving average gamma for learning potential computation",
    )

    # Environment generation arguments
    spade_args.add_argument(
        "--spade-env-temperature",
        type=float,
        default=0.7,
        help="Temperature for environment generation",
    )
    spade_args.add_argument(
        "--spade-env-top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus sampling) for environment generation",
    )
    spade_args.add_argument(
        "--spade-env-top-k",
        type=int,
        default=20,
        help="Top-k sampling for environment generation",
    )
    spade_args.add_argument(
        "--spade-env-generation-template",
        type=str,
        default="qwen3_game_generation",
        help="Template name for environment generation",
    )

    spade_args.add_argument(
        "--spade-env-max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for environment generation",
    )

    # Actor gameplay arguments
    spade_args.add_argument(
        "--spade-actor-temperature",
        type=float,
        default=1.0,
        help="Temperature for actor gameplay",
    )
    spade_args.add_argument(
        "--spade-actor-top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus sampling) for actor gameplay",
    )
    spade_args.add_argument(
        "--spade-actor-top-k",
        type=int,
        default=20,
        help="Top-k sampling for actor gameplay",
    )
    spade_args.add_argument(
        "--spade-actor-template",
        type=str,
        default="qwen3_game",
        help="Template name for actor gameplay",
    )
    spade_args.add_argument(
        "--spade-env-enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for env generation (proposer). "
             "Passes enable_thinking=True to chat template for env role only.",
    )
    spade_args.add_argument(
        "--spade-actor-enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking mode for the actor in multi-turn games. "
             "Each turn's prompt is re-rendered from the canonical (think-"
             "stripped) chat history, the actor generates <think>...</think> "
             "before its answer, and every turn is emitted as its own per-"
             "turn training sample (slime TurnRecord/fan-out machinery); all "
             "turn samples of one episode share a group_id and the episode "
             "reward. Default off = legacy accumulated single-sequence path.",
    )
    spade_args.add_argument(
        "--spade-env-repair-turns",
        type=int,
        default=0,
        help="Multi-turn game-gen repair (Option A). 0 = legacy independent "
             "re-sampling. N>0 = feed the validation error back and ask the "
             "proposer to fix it (inference-only, up to N turns) so the actor "
             "gets a valid env; the proposer is trained on its TURN-1 generation "
             "(valid->regret, broken->0).",
    )
    spade_args.add_argument(
        "--spade-persist-rejected",
        action="store_true",
        default=False,
        help="Persist validation-failed game generations to "
             "<games_dir>/rejected/ for inspection (otherwise discarded).",
    )
    spade_args.add_argument(
        "--spade-max-turns",
        type=int,
        default=50,
        help="Maximum number of turns per game",
    )
    spade_args.add_argument(
        "--spade-actor-max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens for actor gameplay",
    )

    spade_args.add_argument(
        "--spade-use-solver-variance-reward",
        action="store_true",
        default=False,
        help="Enable solver variance reward (reward for solver variance)",
    )
    spade_args.add_argument(
        "--spade-max-context-length",
        type=int,
        default=32768,
        help="Maximum context length for game history",
    )

    # Reward computation arguments
    spade_args.add_argument(
        "--spade-gamma",
        type=float,
        default=0.99,
        help="Discount factor for reward computation",
    )
    spade_args.add_argument(
        "--spade-use-format-reward",
        action="store_true",
        help="Enable spurious format reward (reward for \\boxed{} format)",
    )
    spade_args.add_argument(
        "--spade-format-reward-value",
        type=float,
        default=1.0,
        help="Reward value for using correct format",
    )

    # Game generation arguments
    spade_args.add_argument(
        "--spade-game-regeneration-interval",
        type=int,
        default=50,
        help="Rollout interval between game regeneration (0 to disable)",
    )
    spade_args.add_argument(
        "--spade-num-games-per-rollout",
        type=int,
        default=32,
        help="Number of games to generate per rollout",
    )
    spade_args.add_argument(
        "--spade-games-dir",
        type=str,
        default="./spade_games",
        help="Directory to store generated games",
    )
    spade_args.add_argument(
        "--spade-game-difficulty",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Difficulty level for generated games",
    )
    spade_args.add_argument(
        "--spade-game-type",
        type=str,
        default="cognitive",
        choices=["cognitive", "tool_use"],
        help="Type of games to generate: 'cognitive' (default, logic/math games), "
             "'tool_use' (API/data tool-calling tasks using ToolUseBaseEnv)",
    )
    spade_args.add_argument(
        "--spade-skills",
        type=str,
        nargs="+",
        default=None,
        help="List of skills to use for game generation. "
             "For cognitive games: 'Pattern Recognition', 'Mathematical Reasoning', etc. "
             "For tool_use games: 'Information Retrieval', 'Data Analysis', etc. "
             "Default: top-2 skills for the selected game type.",
    )
    spade_args.add_argument(
        "--spade-skills-per-regen",
        type=int,
        default=0,
        help="Round-robin only this many of the configured --spade-skills per regen epoch "
             "(0 or >= #skills = use all skills every regen). Rotates a disjoint window of "
             "skills each regen so per-rollout cost stays flat while skill breadth scales; "
             "each active skill gets more games and idle gaps are bounded.",
    )
    spade_args.add_argument(
        "--spade-trajectories-per-game",
        type=int,
        default=1,
        help="Number of trajectories to collect per game",
    )
    spade_args.add_argument(
        "--spade-cache-dir",
        type=str,
        default=None,
        help="Directory to cache generated games (e.g., ./cached_games/). "
             "When set, games are saved to cache_dir/step_{rollout_id}/ on creation, "
             "allowing inspection even after regeneration deletes them from games_dir.",
    )

    # Environment reward scaling arguments
    spade_args.add_argument(
        "--spade-env-reward-scaling-variant",
        type=int,
        default=1,
        choices=[0, 1],
        help="Variant for env reward scaling: 0=no reweighting, 1=simple scaling",
    )
    spade_args.add_argument(
        "--spade-max-env-reward-scale",
        type=float,
        default=50.0,
        help="Maximum scale factor for environment rewards (caps the auto-computed scale)",
    )
    spade_args.add_argument(
        "--spade-auto-compute-env-reward-scale",
        action="store_true",
        default=True,
        help="Auto-compute env reward scale from trajectory counts and regeneration interval",
    )
    spade_args.add_argument(
        "--spade-no-auto-compute-env-reward-scale",
        action="store_false",
        dest="spade_auto_compute_env_reward_scale",
        help="Disable auto-compute of env reward scale",
    )
    spade_args.add_argument(
        "--spade-train-on-env-trajectories",
        action="store_true",
        default=True,
        help="Include environment trajectories in training batch (default: True)",
    )
    spade_args.add_argument(
        "--spade-no-train-on-env-trajectories",
        action="store_false",
        dest="spade_train_on_env_trajectories",
        help="Exclude environment trajectories from training batch (for ablation)",
    )

    # Delayed proposer training
    spade_args.add_argument(
        "--spade-proposer-training-delay",
        type=int,
        default=0,
        help="Delay proposer training by N rollout steps. 0=immediate (default). "
             "Env trajectories from step K are buffered and trained at step K+N. "
             "Requires --use-tis for importance ratio correction.",
    )

    # Actor reward normalization arguments
    spade_args.add_argument(
        "--spade-reward-normalization",
        type=str,
        default="ema_baseline",
        choices=["ema_baseline", "grpo"],
        help="Actor reward normalization: 'ema_baseline' (subtract per-game EMA mean) or 'grpo' (per-game z-score)",
    )
    spade_args.add_argument(
        "--spade-game-baseline-decay",
        type=float,
        default=0.5,
        help="Decay rate for per-game baseline EMA (ema_baseline mode only, default: 0.5)",
    )

    # Self-judge arguments (for verifier robustness)
    spade_args.add_argument(
        "--spade-use-self-judge",
        action="store_true",
        default=False,
        help="Enable self-judge to validate generated environments",
    )
    spade_args.add_argument(
        "--spade-self-judge-temperature",
        type=float,
        default=0.3,
        help="Temperature for self-judge (low for deterministic judgments)",
    )
    spade_args.add_argument(
        "--spade-self-judge-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for self-judge response",
    )
    spade_args.add_argument(
        "--spade-self-judge-penalty",
        type=float,
        default=-0.5,
        help="Penalty applied to env reward if self-judge says 'no'",
    )
    spade_args.add_argument(
        "--spade-self-judge-max-turns-to-show",
        type=int,
        default=5,
        help="Max turns to include in trajectory for self-judge evaluation",
    )

    # Environment reward variant arguments
    spade_args.add_argument(
        "--spade-env-reward-variant",
        type=str,
        default="learning_potential",
        choices=["learning_potential", "regret_based", "micro_lp", "blend"],
        help="Environment reward variant: 'learning_potential' (LP-based), 'regret_based' (hint-based regret), "
             "'micro_lp' (game-level improvement within delay window, requires delay > 0), or 'blend' "
             "(matched-timing regret + micro_lp, fixed-scale normalized then weighted; requires delay > 0)",
    )
    spade_args.add_argument(
        "--spade-regret-weight",
        type=float,
        default=0.5,
        help="[blend] weight on the (scale-normalized) matched-timing regret component",
    )
    spade_args.add_argument(
        "--spade-micro-lp-weight",
        type=float,
        default=0.5,
        help="[blend] weight on the (scale-normalized) micro-LP (actor improvement) component",
    )
    spade_args.add_argument(
        "--spade-regret-scale",
        type=float,
        default=0.15,
        help="[blend] fixed nominal scale the regret component is divided by before weighting "
             "(NOT a per-batch std; keeps weights interpretable and avoids small-std blowup)",
    )
    spade_args.add_argument(
        "--spade-micro-lp-scale",
        type=float,
        default=0.10,
        help="[blend] fixed nominal scale the micro-LP component is divided by before weighting",
    )
    spade_args.add_argument(
        "--spade-micro-lp-unsigned",
        action="store_true",
        default=False,
        help="[blend] clamp micro-LP to max(0, late-early) instead of the signed default. "
             "Signed (default) lets a game that made the actor WORSE be a negative signal.",
    )
    spade_args.add_argument(
        "--spade-micro-lp-estimator",
        type=str,
        default="slope",
        choices=["slope", "twobucket"],
        help="[blend] micro-LP estimator. 'slope' (default): WLS slope of per-step "
             "win-rate across the full window (uses every step, ~2-3x lower variance). "
             "'twobucket': legacy late_mean - early_mean.",
    )
    spade_args.add_argument(
        "--spade-frontier-weight",
        type=float,
        default=0.0,
        help="[blend] weight on the frontier anchor: subtract frontier_weight*"
             "((win_rate-0.5)^2 / frontier_scale) from blend. >0 holds the curriculum "
             "near 50% win-rate (max learnability) and breaks the difficulty-easing "
             "death spiral. 0 disables.",
    )
    spade_args.add_argument(
        "--spade-frontier-scale",
        type=float,
        default=0.08,
        help="[blend] fixed nominal scale the frontier penalty (win_rate-0.5)^2 is "
             "divided by before weighting. Tune so comp_frontier_norm matches the "
             "regret/micro-LP component magnitudes.",
    )
    spade_args.add_argument(
        "--spade-plateau-weight",
        type=float,
        default=0.0,
        help="[blend] weight on the flat-top difficulty reward plateau_reward(win_rate). "
             ">0 ADDS a non-negative anchor that is 1.0 in-band (~50% win) and 0 at the "
             "extremes — the additive alternative to the frontier penalty. Intended use: "
             "--spade-plateau-weight 0.6 --spade-regret-weight 0.4 --spade-regret-floor "
             "with micro-LP and frontier off (0). Difficulty regulates which hardness; "
             "floored regret refines which in-band game is most teachable.",
    )
    spade_args.add_argument(
        "--spade-plateau-lo",
        type=float,
        default=0.4,
        help="[blend] low edge of the plateau flat top (full reward for win_rate>=lo).",
    )
    spade_args.add_argument(
        "--spade-plateau-hi",
        type=float,
        default=0.6,
        help="[blend] high edge of the plateau flat top (full reward for win_rate<=hi).",
    )
    spade_args.add_argument(
        "--spade-plateau-ramp",
        type=float,
        default=0.25,
        help="[blend] linear ramp width on each side of the flat top; plateau hits 0 at "
             "lo-ramp and hi+ramp (defaults => 0 at win_rate 0.15 / 0.85).",
    )
    spade_args.add_argument(
        "--spade-regret-floor",
        action="store_true",
        default=False,
        help="[blend] floor the (scaled) regret component at 0 (max(0,.) then clip to 1). "
             "Neutralizes hint-play-timeout / misleading-hint negatives without masking, "
             "and keeps regret on the same [0,1] scale as the plateau. Pair with "
             "--spade-plateau-weight so the whole blend is >=0 (no masking, no death-spiral).",
    )
    spade_args.add_argument(
        "--spade-hint-mode",
        type=str,
        default="self",
        choices=["self", "external"],
        help="Hint generation mode: 'self' (training model) or 'external' (OpenAI/OpenRouter API)",
    )
    spade_args.add_argument(
        "--spade-hint-model",
        type=str,
        default="gpt-5.1-mini",
        help="Model for hint generation in external mode (e.g., openai/gpt-4.1-mini)",
    )
    spade_args.add_argument(
        "--spade-hint-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for hint model API key",
    )
    spade_args.add_argument(
        "--spade-hint-api-base-url",
        type=str,
        default=None,
        help="API base URL for hint model (None for OpenAI default)",
    )
    spade_args.add_argument(
        "--spade-hint-temperature",
        type=float,
        default=0.3,
        help="Temperature for hint generation",
    )
    spade_args.add_argument(
        "--spade-hint-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for hint model response",
    )
    spade_args.add_argument(
        "--spade-hint-plays-per-game",
        type=int,
        default=4,
        help="Number of times to play each game with hint (for regret computation)",
    )
    spade_args.add_argument(
        "--spade-compact-filter",
        action="store_true",
        default=False,
        help="Zero loss_mask on truncated (max-turns) trajectories so they carry no "
             "gradient: running out of turn budget is not a policy failure.",
    )

    # Environment memory arguments (memory-augmented generation)
    spade_args.add_argument(
        "--spade-use-env-memory",
        action="store_true",
        default=False,
        help="Enable environment memory buffer for memory-augmented generation. "
             "High-regret past environments are used as few-shot seeds.",
    )
    spade_args.add_argument(
        "--spade-env-memory-max-size",
        type=int,
        default=200,
        help="Maximum number of environments to keep in memory buffer",
    )

    # Environment validator arguments (rejection sampling during game generation)
    spade_args.add_argument(
        "--spade-use-env-validator",
        action="store_true",
        default=False,
        help="Enable environment validation via LLM rejection sampling during game generation",
    )
    spade_args.add_argument(
        "--spade-env-validator-model",
        type=str,
        default="self",
        help='Model for environment validation. "self" uses the training model (no API key needed). '
             'Otherwise specify a model ID (e.g., google/gemini-3-flash-preview for OpenRouter)',
    )
    spade_args.add_argument(
        "--spade-env-validator-api-key-env",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable name for validator API key",
    )
    spade_args.add_argument(
        "--spade-env-validator-api-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="API base URL for environment validator",
    )
    spade_args.add_argument(
        "--spade-env-validator-temperature",
        type=float,
        default=0.3,
        help="Temperature for environment validator (low for deterministic)",
    )
    spade_args.add_argument(
        "--spade-env-validator-max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for environment validator response",
    )

    # Weave tracking (optional)
    spade_args.add_argument(
        "--spade-use-weave",
        action="store_true",
        help="Enable Weave tracing for experiments",
    )
    spade_args.add_argument(
        "--spade-wandb-project",
        type=str,
        default="spade",
        help="W&B project name for Weave tracking",
    )

    # Fixed model evaluation arguments
    spade_args.add_argument(
        "--spade-fixed-eval-interval",
        type=int,
        default=0,
        help="Rollouts between fixed model evaluations (0=disabled)",
    )
    spade_args.add_argument(
        "--spade-fixed-eval-model",
        type=str,
        default="gpt-5-mini",
        help="Model ID for fixed evaluation (e.g., gpt-5-mini for OpenAI)",
    )
    spade_args.add_argument(
        "--spade-fixed-eval-api-base-url",
        type=str,
        default=None,
        help="API base URL for fixed evaluation (None for OpenAI default)",
    )
    spade_args.add_argument(
        "--spade-fixed-eval-api-key-env",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for API key (default: OPENAI_API_KEY)",
    )
    spade_args.add_argument(
        "--spade-fixed-eval-plays-per-game",
        type=int,
        default=None,
        help="Number of times fixed model plays each game (default: uses --spade-trajectories-per-game)",
    )
    spade_args.add_argument(
        "--spade-fixed-eval-max-concurrent",
        type=int,
        default=128,
        help="Maximum concurrent game plays for rate limiting",
    )
    spade_args.add_argument(
        "--spade-fixed-eval-temperature",
        type=float,
        default=0.7,
        help="Temperature for fixed model generation",
    )
    spade_args.add_argument(
        "--spade-fixed-eval-max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for fixed model response",
    )

    # GEM evaluation
    gem_args = parser.add_argument_group("SPADE GEM Evaluation")

    gem_args.add_argument(
        "--spade-gem-eval-config",
        type=str,
        default=None,
        help="Path to GEM evaluation YAML config file. "
             "Defines tasks with per-task episodes, max_turns, temperature, max_tokens.",
    )

    # Fixed environments
    fixed_env_args = parser.add_argument_group("SPADE Fixed Environment")

    fixed_env_args.add_argument(
        "--spade-mode",
        type=str,
        default="self_play",
        choices=["self_play", "fixed_env"],
        help='Training mode: "self_play" (default, generate games) or "fixed_env" (use external envs)',
    )
    fixed_env_args.add_argument(
        "--spade-fixed-env-source",
        type=str,
        nargs="+",
        default=None,
        help='Environment sources for fixed_env mode. '
             'Formats: "rlve:Sorting", "rlve:*", "gem:game:Sudoku-v0-easy", "gem:*", '
             '"game_file:/path/to/dir", "envduels:/path/to/export"',
    )
    fixed_env_args.add_argument(
        "--spade-envduels-ids-file", default=None,
        help="Optional text file of manifest IDs, one per line; default uses all environments, ignoring export splits.",
    )
    fixed_env_args.add_argument(
        "--spade-envduels-seed", type=int, default=42,
        help="Seed for EnvDuels problem sampling (fresh runs).",
    )
    fixed_env_args.add_argument(
        "--spade-fixed-env-same-problem",
        action="store_true",
        default=False,
        help="All trajectories-per-game plays of an env share ONE generated problem "
             "(per-prompt GRPO grouping, matching RLVE's n-samples-per-prompt). "
             "Requires adapter support; adapters without it fall back to "
             "independent problems.",
    )
    fixed_env_args.add_argument(
        "--spade-difficulty-variant",
        type=str,
        default="sliding_window",
        choices=["sliding_window", "lp"],
        help='Difficulty control variant: "sliding_window" (RLVE-style) or "lp" (LP-based)',
    )
    fixed_env_args.add_argument(
        "--spade-difficulty-tau-acc",
        type=float,
        default=0.9,
        help="Success rate threshold for sliding window promotion (default: 0.9)",
    )
    fixed_env_args.add_argument(
        "--spade-difficulty-tau-num",
        type=int,
        default=8,
        help="Minimum attempts before sliding window promotion check (default: 8)",
    )
    fixed_env_args.add_argument(
        "--spade-difficulty-d-delta",
        type=int,
        default=4,
        help="Maximum sliding window width (default: 4)",
    )
    fixed_env_args.add_argument(
        "--spade-lp-gamma-fast",
        type=float,
        default=0.35,
        help="Fast EMA gamma for LP difficulty controller (default: 0.35)",
    )
    fixed_env_args.add_argument(
        "--spade-lp-gamma-slow",
        type=float,
        default=0.15,
        help="Slow EMA gamma for LP difficulty controller (default: 0.15)",
    )

    # Fixed pool ablation (no adaptive difficulty)
    fixed_env_args.add_argument(
        "--spade-fixed-pool-size",
        type=int,
        default=0,
        help="Fixed pool size. 0 = disabled (use adaptive difficulty). "
             "When > 0, pre-samples a fixed pool of (env_id, difficulty) pairs at init "
             "and uses them for the entire run instead of adaptive difficulty control.",
    )
    fixed_env_args.add_argument(
        "--spade-fixed-pool-seed",
        type=int,
        default=42,
        help="RNG seed for reproducible fixed pool sampling (default: 42)",
    )
    fixed_env_args.add_argument(
        "--spade-fixed-pool-max-difficulty",
        type=int,
        default=5,
        help="Maximum difficulty cap when sampling fixed pool entries (default: 5)",
    )

    fixed_env_args.add_argument(
        "--spade-no-replacement",
        action="store_true",
        default=False,
        help="Sample environments without replacement. Once the pool is exhausted, "
             "refill and reshuffle. Ensures each game is seen before any is repeated.",
    )
    fixed_env_args.add_argument(
        "--spade-static-game-pool",
        action="store_true",
        default=False,
        help="Use pre-generated files from --spade-games-dir without generating "
             "replacement games. This keeps the normal native tool-use rollout path.",
    )

    # Pre-generation of synthetic games (for no-proposer ablation)
    fixed_env_args.add_argument(
        "--spade-pre-generate-games",
        type=int,
        default=0,
        help="Number of synthetic games to pre-generate at init (0=disabled). "
             "When > 0, generates N games using the model at init, saves them, "
             "and creates a SyntheticGameAdapter for fixed-env training.",
    )
    fixed_env_args.add_argument(
        "--spade-pre-generate-difficulty",
        type=str,
        default="hard",
        choices=["easy", "medium", "hard"],
        help="Difficulty level for pre-generated games (default: hard)",
    )
    fixed_env_args.add_argument(
        "--spade-pre-generate-skills",
        type=str,
        nargs="+",
        default=None,
        help="Skills for pre-generated games. Default: Pattern Recognition, Mathematical Reasoning",
    )

    # Corpus-grounded generation
    corpus_args = parser.add_argument_group("SPADE Corpus-Grounded Generation")
    corpus_args.add_argument(
        "--spade-corpus-file",
        type=str,
        default=None,
        help="Path to JSONL corpus file for corpus-grounded game generation. "
             "Each line: {\"text\": \"...\"} or {\"orig_doc\": \"...\"}. "
             "When set, games are generated from sampled corpus documents.",
    )
    corpus_args.add_argument(
        "--spade-corpus-max-doc-tokens",
        type=int,
        default=6000,
        help="Maximum number of tokens per corpus document (word-level approximation, truncated if longer). Default: 6000.",
    )
    corpus_args.add_argument(
        "--spade-corpus-seed",
        type=int,
        default=42,
        help="Random seed for corpus document sampling. Default: 42.",
    )

    # Action format for different environment types
    action_args = parser.add_argument_group("SPADE action format")
    action_args.add_argument(
        "--spade-action-format",
        type=str,
        default="boxed",
        choices=["boxed", "tool_call"],
        help=(
            "Action format the Reasoning Agent uses in responses. "
            "'boxed' = \\boxed{answer} (cognitive games, default). "
            "'tool_call' = <tool_call>JSON</tool_call> (tool-use). "
            "Default: boxed."
        ),
    )

    # Override Slime defaults for SPADE runs
    parser.set_defaults(wandb_always_use_train_step=True)

    return parser
