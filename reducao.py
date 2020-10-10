import io
import random
from flask import Flask, Response, render_template, request, redirect, session, flash, url_for
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG

from matplotlib.figure import Figure

app = Flask(__name__)


@app.route("/")
def index():
    """ Returns html with the img tag for your plot.
    """
    num_x_points = int(request.args.get("num_x_points", 50))
    # in a real app you probably want to use a flask template.
    # return f"""
    # <h1>Flask and matplotlib</h1>
    # <h2>Random data with num_x_points={num_x_points}</h2>
    # <form method=get action="/">
    #   <input name="num_x_points" type=number value="{num_x_points}" />
    #   <input type=submit value="update graph">
    # </form>
    # <h3>Plot as a png</h3>
    # <img src="/matplot-as-image-{num_x_points}.png"
    #      alt="random points as png"
    #      height="200"
    # >
    # <h3>Plot as a SVG</h3>
    # <img src="/matplot-as-image-{num_x_points}.svg"
    #      alt="random points as svg"
    #      height="200"
    # >
    # """
    return render_template("main.html", num_x_points=num_x_points)


@app.route("/matplot-as-image-<int:num_x_points>.png")
def plot_png(num_x_points=50):
    """ renders the plot on the fly.
    """
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x_points = range(num_x_points)
    axis.plot(x_points, [random.randint(1, 30) for x in x_points])

    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")


@app.route("/matplot-as-image-<int:num_x_points>.svg")
def plot_svg(num_x_points=50):
    """ renders the plot on the fly.
    """
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    x_points = range(num_x_points)
    axis.plot(x_points, [random.randint(1, 30) for x in x_points])

    output = io.BytesIO()
    FigureCanvasSVG(fig).print_svg(output)
    return Response(output.getvalue(), mimetype="image/svg+xml")


if __name__ == "__main__":
    app.run(debug=True)