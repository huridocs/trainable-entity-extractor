from transformers import EarlyStoppingCallback, IntervalStrategy


class EarlyStoppingAfterInitialTraining(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if state.global_step < 1000:
            return
        return super().on_evaluate(args, state, control, metrics, **kwargs)
