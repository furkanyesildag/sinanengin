#!/usr/bin/env bash
# =============================================================================
# V10 Paper Bot — Oracle Cloud (veya herhangi bir Ubuntu VM) tek komut kurulum.
#
# Kullanim (VM'e SSH ile girdikten sonra):
#   curl -fsSL https://raw.githubusercontent.com/furkanyesildag/sinanengin/main/deploy/setup.sh | bash
#
# Ne yapar: python+git kurar, repo'yu ceker, venv+bagimliliklar, systemd servisi
# olusturur ve baslatir. Bot --loop ile her 3dk (mum kapanisinda) calisir,
# cokerse otomatik yeniden baslar, VM reboot olsa acilista devreye girer.
# =============================================================================
set -euo pipefail

REPO="https://github.com/furkanyesildag/sinanengin.git"
DIR="$HOME/sinanengin"
USER_NAME="$(whoami)"

echo "==> 1/5 Sistem paketleri (python, venv, git)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3 python3-venv python3-pip git

echo "==> 2/5 Repo (${DIR})"
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" pull --ff-only
else
  git clone --depth 1 "$REPO" "$DIR"
fi

echo "==> 3/5 Sanal ortam + bagimliliklar"
python3 -m venv "$DIR/.venv"
"$DIR/.venv/bin/pip" install --quiet --upgrade pip
"$DIR/.venv/bin/pip" install --quiet -r "$DIR/requirements.txt"

echo "==> 4/5 systemd servisi"
sudo tee /etc/systemd/system/paperbot.service >/dev/null <<UNIT
[Unit]
Description=SE NFT V10 Paper Trading Bot (--loop)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${USER_NAME}
WorkingDirectory=${DIR}
ExecStart=${DIR}/.venv/bin/python3 ${DIR}/paper_bot.py --loop
Restart=always
RestartSec=15
StandardOutput=append:${DIR}/results/service.log
StandardError=append:${DIR}/results/service.log

[Install]
WantedBy=multi-user.target
UNIT

echo "==> 5/5 Servisi baslat (temiz state ile)"
mkdir -p "$DIR/results"
# Repo'dan gelen (GitHub botunun) state'ini sil -> VM sifirdan warmup yapsin
rm -f "$DIR/results/paper_state.json" "$DIR/results/paper_log.txt"
sudo systemctl daemon-reload
sudo systemctl enable paperbot
sudo systemctl restart paperbot
sleep 2
sudo systemctl --no-pager --lines=8 status paperbot || true

echo ""
echo "============================================================"
echo " KURULUM TAMAM. Bot 7/24 calisiyor."
echo "   Canli log:   tail -f ${DIR}/results/paper_log.txt"
echo "   Servis log:  tail -f ${DIR}/results/service.log"
echo "   Durum:       systemctl status paperbot"
echo "   Durdur:      sudo systemctl stop paperbot"
echo "============================================================"
