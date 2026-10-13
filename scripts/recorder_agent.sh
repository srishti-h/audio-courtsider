#!/bin/zsh
# Run the live radio recorder in the background whenever you're logged in (macOS LaunchAgent).
#   scripts/recorder_agent.sh install    start now and at every login
#   scripts/recorder_agent.sh uninstall  stop and remove
#   scripts/recorder_agent.sh status
# It sleeps unless a La Liga / Spanish-club European match is live; the Mac must be awake.
set -e
LABEL=com.audio-courtsider.recorder
PLIST=~/Library/LaunchAgents/$LABEL.plist
REPO=${0:A:h:h}
case "$1" in
  install)
    mkdir -p ~/Library/LaunchAgents "$REPO/data"
    cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array>
    <string>$(command -v uv)</string><string>run</string><string>--directory</string><string>$REPO</string>
    <string>python</string><string>-m</string><string>courtsider.live.recorder</string><string>--station</string><string>ser</string>
  </array>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>StandardOutPath</key><string>$REPO/data/recorder.log</string>
  <key>StandardErrorPath</key><string>$REPO/data/recorder.err.log</string>
</dict></plist>
PL
    launchctl bootstrap gui/$(id -u) "$PLIST" && echo "installed: $PLIST" ;;
  uninstall)
    launchctl bootout gui/$(id -u)/$LABEL 2>/dev/null || true
    rm -f "$PLIST" && echo "removed" ;;
  status)
    launchctl print gui/$(id -u)/$LABEL 2>/dev/null | grep -E "state|pid" || echo "not installed" ;;
  *) echo "usage: $0 install|uninstall|status" ;;
esac
