import framework
import tasks


def register_args(parser: framework.helpers.ArgumentParser):
    parser.add_argument("-batch_size", default=128)
    parser.add_argument("-lr", default=1e-3)
    parser.add_argument("-wd", default=0.0)
    parser.add_argument("-lr_warmup", default=0)
    parser.add_argument("-test_interval", default=1000)
    parser.add_argument("-state_size", default=128)
    parser.add_argument("-stop_after", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-task", default="trafo_scan")
    parser.add_argument("-dropout", default=0.0)
    parser.add_argument("-grad_clip", default="1.0", parser=parser.float_or_none_parser)
    parser.add_argument("-scan.train_split", default="simple", parser=parser.str_list_parser)
    parser.add_argument("-scan.length_cutoff", default=22)
    parser.add_argument("-layer_sizes", default="800,800,256", parser=parser.int_list_parser)
    parser.add_argument("-transformer.n_heads", default=4)
    parser.add_argument("-transformer.variant", default="scaledinit")
    parser.add_argument("-transformer.ff_multiplier", default=2.0)
    parser.add_argument("-transformer.encoder_n_layers", default=3)
    parser.add_argument("-transformer.decoder_n_layers", default="3", parser=parser.int_or_none_parser)
    parser.add_argument("-transformer.tied_embedding", default=True)
    parser.add_argument("-test_batch_size", default="None", parser=parser.int_or_none_parser)
    parser.add_argument("-dm_math.tasks", default="algebra__linear_1d", parser=parser.str_list_parser)
    parser.add_argument("-dm_math.train_splits", default="easy,medium,hard", parser=parser.str_list_parser)
    parser.add_argument("-lr_sched.steps", default="", parser=parser.int_list_parser)
    parser.add_argument("-lr_sched.gamma", default=0.1)
    parser.add_argument("-lr_sched.type", default="step", choice=["step", "noam"])
    parser.add_argument("-optimizer", default="adam", choice=["adam", "sgd"])
    parser.add_argument("-adam.betas", default="0.9,0.999", parser=parser.float_list_parser)
    parser.add_argument("-amp", default=False)
    parser.add_argument("-cogs.generalization_test_interval", default=2500)
    parser.add_argument("-label_smoothing", default=0.0)
    parser.add_argument("-pcfg.split", default="simple", choice=["simple", "productivity", "substitutivity",
                                                                 "systematicity"])
    parser.add_argument("-cfq.split", default="random", choice=["random", "query_complexity", "question_complexity",
                                                                "query_pattern", "question_pattern", "mcd1", "mcd2", 
                                                                "mcd3"])
    parser.add_argument("-max_length_per_batch", default="none", parser=parser.int_or_none_parser)
    parser.add_argument("-log_sample_level_loss", default=False)

    parser.add_profile([
        parser.Profile("cfq_trafo", {
            "task": "cfq_trafo",
            "transformer.variant": "noscale",
            "state_size": 128,
            "transformer.n_heads": 16,
            "transformer.ff_multiplier": 2,
            "transformer.encoder_n_layers": 2,
            "transformer.decoder_n_layers": 2,
            "grad_clip": 1,
            "stop_after": 50000,
            "dropout": 0.1,
            "batch_size": 512,
            "lr": 1e-4,
        }),

        parser.Profile("cfq_universal_trafo", {
            "transformer.variant": "universal_noscale",
            "state_size": 256,
            "transformer.n_heads": 4,
            "transformer.ff_multiplier": 2,
            "transformer.encoder_n_layers": 6,
            "transformer.decoder_n_layers": 6,
        }, include="cfq_trafo"),

        parser.Profile("cogs_trafo_small", {
            "task": "cogs_transformer",
            "state_size": 512,
            "transformer.n_heads": 4,
            "transformer.ff_multiplier": 1,
            "transformer.encoder_n_layers": 2,
            "transformer.decoder_n_layers": 2,
            "grad_clip": "none",
            "stop_after": 50000,
            "dropout": 0.1,
            "batch_size": 128,
            "lr": 2,
            "lr_sched.type": "noam",
            "lr_warmup": 4000,
        }),

        parser.Profile("deepmind_math", {
            "task": "dm_math_transformer",
            "lr": 1e-4,
            "stop_after": 50000,
            "batch_size": 256,
            "mask_loss_weight": 0.001,
            "state_size": 512,
            "transformer.n_heads": 8,
            "transformer.ff_multiplier": 4,
            "transformer.encoder_n_layers": 6,
            "transformer.decoder_n_layers": 6,
            "test_batch_size": 1024,
            "grad_clip": 0.1
        }),

        parser.Profile("pcfg_trafo", {
            "task": "pcfg_transformer",
            "state_size": 512,
            "transformer.n_heads": 8,
            "transformer.ff_multiplier": 4,
            "transformer.encoder_n_layers": 6,
            "transformer.decoder_n_layers": 6,
            "lr": 1e-3,
            "grad_clip": "1",
            "stop_after": 1000000,
            "batch_size": 64
        }),

        parser.Profile("trafo_scan", {
            "lr": 1e-3,
            "grad_clip": "5",
            "stop_after": 15000,
            "batch_size": 256,
            "dropout": 0.5,
            "embedding_size": 16,
            "task": "trafo_scan",
            "state_size": 128,
            "transformer.n_heads": 8,
            "test_batch_size": 2048
        })
    ])


def main():
    helper = framework.helpers.TrainingHelper(wandb_project_name="modules",
                                              register_args=register_args, extra_dirs=["export", "model_weights"])

    def invalid_task_error(_):
        assert False, f"Invalid task: {helper.args.task}"

    constructors = {
        "pcfg_transformer": tasks.PCFGTransformer,
        "cogs_transformer": tasks.COGSTransformer,
        "trafo_scan": tasks.ScanTransformer,
        "scan_resplit_transformer": tasks.ScanResplitTransformer,
        "cfq_trafo": tasks.CFQTransformer,
        "dm_math_transformer": tasks.DMMathTransformer,
    }

    task = constructors.get(helper.args.task, invalid_task_error)(helper)    
    task.train()
    helper.finish()


if __name__ == "__main__":
    main()
