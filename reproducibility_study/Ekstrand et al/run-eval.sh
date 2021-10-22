#!/bin/bash

export JAVA_HOME="$HOME/opt/java8"
export PATH="$HOME/miniconda2/bin:$JAVA_HOME/bin:$PATH"
export GRADLE_OPTS=-Xmx128m
GRADLE_HOME="/scratch/mekstrand/gradle-homes/$HOSTNAME"

source activate demog

ulimit -v 167772160
ulimit -u 256
ulimit -s 32768
ulimit -a
./gradlew -g "$GRADLE_HOME" --project-cache-dir ".gradle-$HOSTNAME" --no-daemon "$@"
