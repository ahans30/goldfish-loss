
# Goldfish Loss: Mitigating Memorization in Generative LLMs

<p align="center">
  <img src="assets/goldfish-loss.jpg" width="200" height="200" alt="A very smart goldfish">
</p>


#### We introduce goldfish loss — a strikingly simple technique to mitigating extractable memorization in large language models.

## Getting Started

This codebase is setup to (pre)train large language models in a distributed training setup using SLURM. This repo is written and tested on AMD compute-nodes and is a fork of Lightning AI's [LitGPT repository](https://github.com/Lightning-AI/litgpt).

For implementation of goldfish loss, please checkout [`apply_goldfish`](https://github.com/ahans30/goldfish-loss/blob/70bfad87dcf69da2921bcad08e662925ab2ab60b/lit_gpt/utils.py#L241) method in `lit_gpt/utils.py`.

### Installation
Before running below command, checkout the script and setup path variables specific to your compute instance (e.g. `INSTALLDIR`).

```bash
$ bash install.sh
```

This will take some time, but will create a folder called `goldfish_loss` and a folder called `tiny_plugins_rccl` in the installation directory (which is `$HOME` by default). The conda folder contains a fully functional conda environment containing all neccessary packages to run our training scripts. The RCCL folder contains code for the interconnect plugin that is crucial for multi-node jobs. You can enable this environment by using `source activate ${INSTALLDIR}/<PATH-TO-ENV_NAME>`.

Please note, this installation is specific for AMD's compute nodes and is written as such.


## Usage

Below command will invocate a python script that will inturn queue a SLURM job specific to frontier cluster used for this work. You can use `--dryrun` to retrieve the `sbatch` script that can be tuned for other clusters.

```bash
$ launch_scripts/launch_jobs_1b_hashtable.sh
```

You can find the training config in `launch_scripts/config/` in yaml format.


## Contributing

This is WIP repo and we are working on porting all experiments from the paper.

We believe in open source community development. Feel free to add things, open issues, start pull requests, or reach out to us with any thoughts or questions.

## Cite our work

If you find our work useful, please cite our paper:

```bibtex
@misc{hans2024like,
      title={Be like a Goldfish, Don't Memorize! Mitigating Memorization in Generative LLMs}, 
      author={Abhimanyu Hans and Yuxin Wen and Neel Jain and John Kirchenbauer and Hamid Kazemi and Prajwal Singhania and Siddharth Singh and Gowthami Somepalli and Jonas Geiping and Abhinav Bhatele and Tom Goldstein},
      year={2024},
      eprint={2406.10209},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```

## Acknowledgements
1. This codebase is developed as a collective effort from [tomg-group-umd](https://github.com/tomg-group-umd) members for LLM (pre)training and is fork of the [LitGPT codebase](https://github.com/Lightning-AI/litgpt).
1. This code is tested on *frontier* resources of the Oak Ridge Leadership Computing Facility at the Oak Ridge National Laboratory which is supported by the Office of Science of the U.S. Department of Energy under Contract No. DE-AC05-00OR22725. This project is supported by the award for computer time was provided by the U.S. Department of Energy’s (DOE) Innovative and Novel Computational Impact on Theory and Experiment (INCITE) Program. Financial support was provided by the ONR MURI program and the AFOSR MURI program. Private support was provided by Capital One Bank, the Amazon Research Award program, and Open Philanthropy. Further support was provided by the National Science Foundation (IIS-2212182), and by the NSF TRAILS Institute (2229885).