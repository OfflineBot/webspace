FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY . /usr/share/nginx/html

# precompressed next to the original, so gzip_static answers without
# re-compressing 53 MB of wasm on every request
RUN gzip -9k /usr/share/nginx/html/game_bg.wasm
