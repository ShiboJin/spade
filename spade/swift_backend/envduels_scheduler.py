"""Stop Gym rollouts at context limits, preserving per-round generation."""
from copy import copy, deepcopy
import os

from swift.rollout.multi_turn import GYMScheduler, multi_turns
from swift.utils import get_logger


logger = get_logger()


class EnvDuelsScheduler(GYMScheduler):
    def __init__(self, *args, template=None, **kwargs):
        super().__init__(*args, **kwargs)
        if template is None:
            template = self.infer_engine.template
        self.context_limit = template.max_length
        self.inference_template = copy(template)
        self.inference_template.set_mode("vllm")
        # Count the complete next prompt rather than its truncated encoding.
        self.inference_template.max_length = None

    async def on_turn_end(self, infer_request, response_choice, current_turn):
        result = await super().on_turn_end(infer_request, response_choice, current_turn)
        if os.environ.get("RANK", "0") == "0":
            logger.info("EnvDuels rollout %s: turn=%s reward=%s env_done=%s",
                        infer_request.uuid, current_turn,
                        result["rollout_infos"].get("total_reward"), result["done"])
        if result["done"]:
            return result

        reason = None
        if self.max_turns and current_turn >= self.max_turns:
            reason = "max_turns"
        elif self.context_limit is not None:
            observation = self._pending_obs.get(infer_request.uuid)
            messages = deepcopy(infer_request.messages)
            if observation is not None:
                messages.append({"role": "user", "content": observation})
            encoded = self.inference_template.encode({"messages": messages})
            if len(encoded["input_ids"]) >= self.context_limit:
                reason = "context_length"
                # Let GRPO's overlong_filter mask this truncated trajectory.
                response_choice.finish_reason = "length"

        if reason is not None:
            result["done"] = True
            result["rollout_infos"].update(gym_truncated=True, truncation_reason=reason)
            await self._close_and_remove(infer_request.uuid)
        return result


multi_turns["envduels_scheduler"] = EnvDuelsScheduler
