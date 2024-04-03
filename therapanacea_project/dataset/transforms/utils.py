import logging
from typing import Union

import torchvision
from omegaconf import OmegaConf
from torchvision.transforms import Compose


def instantiate_transforms_from_config(
        transform_config: dict[str, list[Union[str, dict]]]
    ) -> Compose:
    compositions = []
    try:
       for framework, fn_name, fn_params in zip(
            transform_config["frameworks"],
            transform_config["classes"],
            transform_config["classes_params"],
        ):
            fn_params = OmegaConf.to_object(fn_params)
            if framework == "torchvision":
                fn = getattr(torchvision.transforms, fn_name)(**fn_params)
            #elif framework == "custom":
            #    fn = custom_fn(fn_name)(**fn_params)
            compositions.append(fn)

            logging.info(f"compositions: {compositions}")

    except Exception as e:
        logging.info(f"Got error on {fn_name} with kwargs: {fn_params}")
        raise ValueError(f"Exception: {e}")

    logging.info(f"composition transforms: {compositions}")

    return Compose(compositions)