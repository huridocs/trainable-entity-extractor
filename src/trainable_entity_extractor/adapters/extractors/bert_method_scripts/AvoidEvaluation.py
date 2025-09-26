from transformers import DefaultFlowCallback


class AvoidEvaluation(DefaultFlowCallback):
    def on_step_end(self, args, state, control, **kwargs):
        super().on_step_end(args, state, control, **kwargs)

        if state.global_step <= 1000:
            control.should_evaluate = False

        return control
