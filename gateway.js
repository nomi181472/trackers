const http = require("http");

const PROXY_PORT = process.env.PORT || 7860;
const FRONTEND_PORT = 3000;
const BACKEND_PORT = 8000;

const server = http.createServer((req, res) => {
  const isApi =
    req.url.startsWith("/api") ||
    req.url.startsWith("/docs") ||
    req.url.startsWith("/openapi.json");

  const targetPort = isApi ? BACKEND_PORT : FRONTEND_PORT;

  const options = {
    hostname: "127.0.0.1",
    port: targetPort,
    path: req.url,
    method: req.method,
    headers: req.headers,
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });

  proxyReq.on("error", (err) => {
    if (!res.headersSent) {
      res.writeHead(502, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Service unavailable", detail: err.message }));
    }
  });

  req.pipe(proxyReq);
});

server.listen(PROXY_PORT, "0.0.0.0", () => {
  console.log(`Unified Gateway listening on 0.0.0.0:${PROXY_PORT}`);
});
