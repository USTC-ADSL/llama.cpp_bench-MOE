#!/system/bin/sh
while IFS= read -r line; do
    [ "$line" = "$(printf '\r')" ] && break
done
printf 'HTTP/1.0 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: 8149865792\r\nConnection: close\r\n\r\n'
exec cat /data/local/tmp/qwen1.5-moe-a2.7b-chat-q4_0.gguf
