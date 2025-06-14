#!/bin/bash

PUB_KEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQC91FRucJ8+l1JrFeAfJxYp1e7eamMuPmtwrzdstNrlK+LbJphziN5fMLz6JbNtbPO9+2NiyhZn9ULgTTbjWi8HzavMwZCQckNZPWOOBnMf3VCYg0wKTG7H3Mg0WZkm7L994SASj0FGhln4GqsEsDXykuNr2gY4sHB7TfmqkBExHxENfUYiC7VG1pL/zJEtopE0oPSQVJ3/0Di0TmxSKH1wA3FIywzk8rRASvp78SRB7PaLdlNz3o/mhIbqZ2dzFNu/Y33uwr9FhytHOmd7lybA5qDN26gipMeYUs2gUrzCoSoOz3yX7LYrGg73WbekRrzte4IN9Z53zxwaLaQ/XbGhi/ImflT4qYmPq9QicxyagIHQJ3zhLoSikKRh9vE7dUbfWu53fNZmvb2oDIfwxAUHriFywWAKztLzbBFsgHWA6Wwr67D7hEBYk4pfxF0FNrbVMcO0v0LZoiKdJ8bTsJiF7179daMLi6zFFWnRoS6uumk2LQ8GdpYogfLgVlvjahc= debian@light-node-0"

# ホスト一覧ファイル（1行1IPまたはホスト名）
HOSTFILE="hosts.txt"

while read -r HOST; do
  echo "🔐 Adding key to $HOST..."
  ssh -o StrictHostKeyChecking=no debian@$HOST "mkdir -p ~/.ssh && echo '$PUB_KEY' >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && chmod 700 ~/.ssh"
done < "$HOSTFILE"
