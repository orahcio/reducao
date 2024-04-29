from app import app

if __name__ == "__main__":
    # app.run()
    port = int(os.environ.get("PORT",8001))
    app.run(host="0.0.0.0", port=port)
