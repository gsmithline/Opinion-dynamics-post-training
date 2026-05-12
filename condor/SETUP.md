# HTCondor setup (MPI-IS Tübingen)

One-time setup on `login2`, then `condor_submit_bid` to launch the 5-config LLM sweep.

## 1. Install miniconda (if not already)

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash   # reload shell
```

## 2. Clone repo + create env

```bash
cd ~
git clone <REPO_URL> Opinion-dynamics-post-training
cd Opinion-dynamics-post-training
conda env create -f environment.yml
conda activate opdyn
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

## 3. Wandb key (file-based, per your request)

```bash
echo "<YOUR_WANDB_API_KEY>" > ~/.wandb_key
chmod 600 ~/.wandb_key
```

`run_one.sh` reads this file and exports `WANDB_API_KEY`.

## 4. Pokec data

The sweep expects `pokec_dataset/` inside the repo. If not committed, scp from
your laptop:

```bash
# from laptop
scp -r Opinion-dynamics-post-training/pokec_dataset \
    gsmithline@login2:/home/gsmithline/Opinion-dynamics-post-training/
```

## 5. Smoke test (CPU or GPU login node)

```bash
chmod +x condor/run_one.sh
RETRAIN_T=2 SFT_EPOCHS=1 condor/run_one.sh smoke sft 0.0
```

Kill after one round prints to confirm env + wandb + data all work.

## 6. Submit the sweep

```bash
mkdir -p condor/logs
condor_submit condor/llm_sweep.sub          # bids the minimum (1)
# or, if the queue is busy:
condor_submit_bid 10 condor/llm_sweep.sub
condor_q $USER
```

### Bidding notes (MPI-IS)

- Second-price auction: you are charged the *next* bid in the queue, not your own. Overbidding is safe when the queue is thin.
- Computing units per job in this sub file (4 CPU, 32G RAM, 1 GPU):
  `max(4, 32/16) + 24 = 28 units/hour`.
- Starter salary is 2500/week; one 5-config sweep at 2 hr each costs at most `5 * 28 * 2 * bid` cluster-dollars. Real clearing price is usually much lower.
- Check balance + queue: `https://logger.cluster.is.localnet/htcondor/banking`
- Set auto-kill in the banking portal so a stuck job can't drain the account (default −2× salary is fine).
- Bump bid later for a queued job: `condor_change_bid <bid> <ClusterId[.ProcId]>`.
- Ask Celestine for a salary top-up before larger-model sweeps.

## 7. Monitor

```bash
condor_q $USER                          # status
tail -f condor/logs/llm_sftkl_b1.out    # live stdout
condor_ssh_to_job <CLUSTER_ID>.<PROC>   # attach if needed
```

## 8. Adjusting the sweep

- Rounds: edit `RETRAIN_T=30` in the `environment = ...` line of `llm_sweep.sub`.
- Base model: edit `BASE_MODEL=...` in the same line (e.g. `Qwen/Qwen2.5-1.5B-Instruct`).
- Add/remove configs: edit `condor/configs.txt`.
- Resources: edit `request_cpus / request_memory / request_gpus` in `llm_sweep.sub`.

## 9. Baselines (optional)

Baselines are cheap; run them on the login node or a single condor job rather
than a parallel sweep:

```bash
RETRAIN_T=30 bash run_experiments.sh baselines
```
