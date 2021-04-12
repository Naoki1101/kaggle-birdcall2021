# import os
# import re
# import time
# import shutil
# import dataclasses

# from pathlib import PosixPath

# from .data import DataHandler


# @dataclasses.dataclass
# class Kaggle:
#     compe_name: str
#     run_name: str

#     def __post_init__(self, log_dir: PosixPath, notebook_dir: PosixPath):
#         self.slug = re.sub("[_.]", "-", self.run_name)
#         self.log_dir = log_dir / self.run_name
#         self.notebook_dir = notebook_dir / self.run_name

#         self.dh = DataHandler()

#         account = self.dh.load(const.CONFIG_DIR / "account.yml")
#         self.KAGGLE_USERNAME = account["kaggle"]["username"]

#     def submit(self, comment: str) -> None:
#         cmd = f'kaggle competitions submit -c {self.compe_name} \
#             -f ../data/output/{self.run_name}.csv  -m "{comment}"'
#         self._run(cmd)
#         print(f"\n\nhttps://www.kaggle.com/c/{self.compe_name}/submissions\n\n")

#     def create_dataset(self) -> None:
#         cmd_init = f"kaggle datasets init -p {self.log_dir}"
#         cmd_create = f"kaggle datasets create -p {self.log_dir} -q"
#         self._run(cmd_init)
#         self._insert_dataset_metadata()
#         self._run(cmd_create)

#     def push_notebook(self) -> None:
#         time.sleep(20)
#         self._prepare_notebook_dir()
#         cmd_init = f"kaggle kernels init -p {self.notebook_dir}"
#         cmd_push = f"kaggle kernels push -p {self.notebook_dir}"
#         self._run(cmd_init)
#         self._insert_kernel_metadata()
#         self._run(cmd_push)

#     def _run(self, cmd: str) -> None:
#         os.system(cmd)

#     def _insert_dataset_metadata(self) -> None:
#         metadata_path = f"{self.log_dir}/dataset-metadata.json"
#         meta = self.dh.load(metadata_path)
#         meta["title"] = self.run_name
#         meta["id"] = re.sub("INSERT_SLUG_HERE", f"sub-{self.slug}", meta["id"])
#         self.dh.save(metadata_path, meta)

#     def _insert_kernel_metadata(self) -> None:
#         metadata_path = f"{self.notebook_dir}/kernel-metadata.json"
#         meta = self.dh.load(metadata_path)
#         meta["title"] = self.run_name
#         meta["id"] = re.sub("INSERT_KERNEL_SLUG_HERE", self.slug, meta["id"])
#         meta["code_file"] = f"sub_{self.run_name}.ipynb"
#         meta["language"] = "python"
#         meta["kernel_type"] = "notebook"
#         meta["is_private"] = "true"
#         meta["enable_gpu"] = "true"
#         meta["enable_internet"] = "false"
#         meta["dataset_sources"] = [f"{self.username}/sub-{self.slug}"]
#         meta["competition_sources"] = [f"{self.compe_name}"]
#         meta["kernel_sources"] = []

#         self.dh.save(metadata_path, meta)

# def _prepare_notebook_dir(self) -> None:
#     (const.NOTEBOOK_DIR / self.run_name).mkdir(exist_ok=True)
#     if "resnet" in self.run_name:
#         shutil.copy(
#             "../notebooks/resnet_for_inference.ipynb",
#             f"{self.notebook_dir}/sub_{self.run_name}.ipynb",
#         )
#     elif "se_resnext" in self.run_name:
#         shutil.copy(
#             "../notebooks/se_resnext_for_inference.ipynb",
#             f"{self.notebook_dir}/sub_{self.run_name}.ipynb",
#         )
#     elif "efficientnet" in self.run_name:
#         shutil.copy(
#             "../notebooks/efficientnet_for_inference.ipynb",
#             f"{self.notebook_dir}/sub_{self.run_name}.ipynb",
#         )
#     elif "resnest" in self.run_name:
#         shutil.copy(
#             "../notebooks/resnest_for_inference.ipynb",
#             f"{self.notebook_dir}/sub_{self.run_name}.ipynb",
#         )
