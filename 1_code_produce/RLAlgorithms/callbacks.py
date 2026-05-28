from ray.rllib.algorithms.callbacks import DefaultCallbacks


#At the end of each episodes this logs values of the info dict (from the environment) and tells RLlib to 
#store them as custom metrics such that we can log them 

class RampMeterCallbacks(DefaultCallbacks):

    def on_episode_end(self, *, episode, **kwargs):


        infos = episode._last_infos
        if not infos:
            return

        info = infos.get("agent0", {})

        if not info:
            return

        throughput = info.get("throughput_veh_h")
        speed = info.get("speed_m_s")

        # for RLlib aggregation
        if throughput is not None:
            episode.custom_metrics["throughput_veh_h"] = throughput

        if speed is not None:
            episode.custom_metrics["speed_m_s"] = speed

        # print directly
        print(f"[EPISODE DONE] throughput={throughput:.2f}, speed={speed:.2f}")