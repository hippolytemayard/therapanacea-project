import logging
from typing import Union

import torchmetrics
from omegaconf import OmegaConf
from torchmetrics import MetricCollection


def instantiate_metrics_from_config(
    metrics_config: dict[str, list[Union[str, dict]]]
) -> MetricCollection:
    compositions = []
    try:
        for framework, fn_name, fn_params in zip(
            metrics_config["frameworks"],
            metrics_config["classes"],
            metrics_config["classes_params"],
        ):
            fn_params = OmegaConf.to_object(fn_params)
            if framework == "torchmetrics":
                fn = getattr(torchmetrics.classification, fn_name)(**fn_params)
            # elif framework == "custom":
            #    fn = custom_fn(fn_name)(**fn_params)
            compositions.append(fn)

    except Exception as e:
        logging.info(f"Got error on {fn_name} with kwargs: {fn_params}")
        raise ValueError(f"Exception: {e}")
    
    logging.info(f"composition metrics: {compositions}")

    return MetricCollection(compositions)
