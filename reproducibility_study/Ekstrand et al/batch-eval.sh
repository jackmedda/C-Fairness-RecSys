#!/bin/sh
#SBATCH --exclusive -c28 -N1

. "$HOME/.nvm/nvm.sh"
nvm use lts/boron

export SLACK_WEBHOOK_CLI_URL="https://hooks.slack.com/services/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"
export SLACK_WEBHOOK_CLI_CHANNEL="#eval-issues"
export SLACK_WEBHOOK_CLI_USERNAME="R2"
export SLACK_WEBHOOK_CLI_EMOJI=":r2d2:"

slack-hook send -m ":rocket: <@mdekstrand>, we are starting job $SLURM_JOB_NAME!"

if srun ./run-eval.sh "$@"; then
    slack-hook send -m ":tada: <@mdekstrand> job $SLURM_JOB_NAME completed!"
else
    slack-hook send -m ":skull: <@mdekstrand> job $SLURM_JOB_NAME is not happy"
    exit 2
fi
