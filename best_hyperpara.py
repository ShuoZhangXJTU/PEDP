best_hyperpara = {
    "pedp":
    {
        "gumbel": True,
        "h_dim": 200,
        "plan_gamma": 0.311015333058422,
        "pred_gamma": 12.1184526606073,
        "state_gamma": 0.31349603409602,
        "term_gamma": 0.0411142841203765,
        "tau_plan_a": 1e-3,
        "tau_plan_t": 1e-3,
        "temperature": 1e-3,
        "lr": 0.000785927677712986,
    },
    "seq":
    {
        "batchsz": 32,
        "dropout": 0.1,
        "clip": 0.5,
        "curriculum": 1,
        "h_dim": 200,
        "lr": 0.0001,
    },
    "gcas":
    {
        "batchsz": 32,
        "dropout": 0.,
        "clip": 0.5,
        "curriculum": 1,
        "h_dim": 200,
        "lr": 1e-3,
    },
    "cls":
    {
        "batchsz": 32,
        "dropout": 0.1,
        "plan_gamma": 0,
        "state_gamma": 0,
        "KL_gamma": 0,
        "clip": 10,
        "curriculum": 1,
        "h_dim": 200,
        "lr": 3e-4,
        "search_curriculum": 1,
    },
    "md":
    {
        "gumbel": True,
        "batchsz": 32,
        "dropout": 0.1,
        "plan_gamma": 0,
        "state_gamma": 0,
        "KL_gamma": 0,
        "clip": 10,
        "curriculum": 1,
        "h_dim": 200,
        "lr": 3e-4,
    }
}