#!/bin/bash
# Jetson 종료(전원 끄기) (확인창 후 systemctl poweroff — 로그인 세션 사용자는 암호 불필요)
zenity --question --title "Jetson 종료" \
       --text "Jetson을 종료(전원 끄기)하시겠습니까?" \
       --ok-label "종료" --cancel-label "취소" --width 320 \
  && systemctl poweroff
