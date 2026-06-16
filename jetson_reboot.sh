#!/bin/bash
# Jetson 재시작 (확인창 후 systemctl reboot — 로그인 세션 사용자는 암호 불필요)
zenity --question --title "Jetson 재시작" \
       --text "Jetson을 재시작하시겠습니까?" \
       --ok-label "재시작" --cancel-label "취소" --width 320 \
  && systemctl reboot
