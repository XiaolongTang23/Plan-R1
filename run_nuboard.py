import logging
import os
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.nuboard.nuboard import NuBoard
from nuplan.planning.script.builders.scenario_building_builder import (
    build_scenario_builder,
)
from nuplan.planning.script.builders.utils.utils_config import update_config_for_nuboard
from datasets import set_default_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()


def initialize_nuboard(cfg: DictConfig) -> NuBoard:
    """
    Sets up dependencies and instantiates a NuBoard object.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    :return: NuBoard object.
    """
    # Update and override configs for nuboard
    update_config_for_nuboard(cfg=cfg)

    scenario_builder = build_scenario_builder(cfg)

    # Build vehicle parameters
    vehicle_parameters: VehicleParameters = instantiate(
        cfg.scenario_builder.vehicle_parameters
    )
    profiler_path = None
    if cfg.profiler_path:
        profiler_path = Path(cfg.profiler_path)

    nuboard = NuBoard(
        profiler_path=profiler_path,
        nuboard_paths=cfg.simulation_path,
        scenario_builder=scenario_builder,
        port_number=cfg.port_number,
        resource_prefix=cfg.resource_prefix,
        vehicle_parameters=vehicle_parameters,
    )

    return nuboard


@hydra.main(config_path="./config", config_name="default_nuboard")
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario.
    :param cfg: DictConfig. Configuration that is used to run the experiment.
    """
    nuboard = initialize_nuboard(cfg)
    nuboard.run()


if __name__ == "__main__":
    main()