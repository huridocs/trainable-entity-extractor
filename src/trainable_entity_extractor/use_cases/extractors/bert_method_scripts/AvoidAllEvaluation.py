from transformers import DefaultFlowCallback


class AvoidAllEvaluation(DefaultFlowCallback):
    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)

        if state.global_step <= 8000:
            control.should_evaluate = False

        return control
