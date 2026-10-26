import os
import sys
import traceback

# 1. Inject bundled lib directory into sys.path before any third-party imports
current_dir = os.path.abspath(os.path.dirname(__file__))
lib_dir = os.path.join(current_dir, "lib")
if os.path.exists(lib_dir) and lib_dir not in sys.path:
    sys.path.insert(0, lib_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 2. Resolve port assigned by Zoho Catalyst AppSail
port_str = os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT", os.environ.get("PORT", "9000"))
try:
    port = int(port_str)
except ValueError:
    port = 9000

print(f"[AppSail] Booting on port {port} (Python {sys.version})")

try:
    import uvicorn
    from app.main import app
    
    if __name__ == "__main__":
        print(f"[AppSail] Starting Uvicorn on 0.0.0.0:{port}...")
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

except Exception as err:
    # If any module import or startup fails, run diagnostic HTTP server so curl returns the exact error
    err_trace = traceback.format_exc()
    print(f"[AppSail CRITICAL ERROR]: {err_trace}", file=sys.stderr)
    
    from http.server import HTTPServer, BaseHTTPRequestHandler
    
    class DiagnosticHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"AppSail Diagnostic Traceback:\n\n{err_trace}".encode("utf-8"))

    server = HTTPServer(("0.0.0.0", port), DiagnosticHandler)
    print(f"[AppSail] Diagnostic server listening on {port}")
    server.serve_forever()
