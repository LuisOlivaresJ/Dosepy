from matplotlib import cbook
from matplotlib.widgets import Widget


class MultiCursor(Widget):
    """
    Provide a vertical (default) and/or horizontal line cursor shared between
    multiple Axes.

    For the cursor to remain responsive you must keep a reference to it.

    Based from matplotlib.widgets.Multicursor (v3.8.0)

    Parameters
    ----------
    canvas : object
        This parameter is entirely unused and only kept for back-compatibility.

    axes : list of `~matplotlib.axes.Axes`
        The `~.axes.Axes` to attach the cursor to.

    useblit : bool, default: True
        Use blitting for faster drawing if supported by the backend.
        See the tutorial :ref:`blitting`
        for details.

    horizOn : bool, default: False
        Whether to draw the horizontal line.

    vertOn : bool, default: True
        Whether to draw the vertical line.

    Other Parameters
    ----------------
    **lineprops
        `.Line2D` properties that control the appearance of the lines.
        See also `~.Axes.axhline`.
    """

    def __init__(self, canvas, axes, *, useblit=True, horizOn=True, vertOn=True,
                 **lineprops):
        # canvas is stored only to provide the deprecated .canvas attribute;
        # once it goes away the unused argument won't need to be stored at all.
        self._canvas = canvas

        self.axes = axes
        self.horizOn = horizOn
        self.vertOn = vertOn

        self._canvas_infos = {
            ax.figure.canvas: {"cids": [], "background": None} for ax in axes}

        xmin, xmax = axes[-1].get_xlim()
        ymin, ymax = axes[-1].get_ylim()
        xmid = 0.5 * (xmin + xmax)
        ymid = 0.5 * (ymin + ymax)

        self.visible = True
        self.useblit = (
            useblit
            and all(canvas.supports_blit for canvas in self._canvas_infos))

        if self.useblit:
            lineprops['animated'] = True

        self.vlines = [ax.axvline(xmid, visible=False, **lineprops)
                       for ax in axes]
        self.hlines = [ax.axhline(ymid, visible=False, **lineprops)
                       for ax in axes]

        self.connect()

    #needclear = _api.deprecated("3.7")(lambda self: False)

    def connect(self):
        """Connect events."""
        for canvas, info in self._canvas_infos.items():
            info["cids"] = [
                canvas.mpl_connect('motion_notify_event', self.onmove),
                canvas.mpl_connect('draw_event', self.clear),
            ]

    def disconnect(self):
        """Disconnect events."""
        for canvas, info in self._canvas_infos.items():
            for cid in info["cids"]:
                canvas.mpl_disconnect(cid)
            info["cids"].clear()

    def clear(self, event):
        """Clear the cursor."""
        if self.ignore(event):
            return
        if self.useblit:
            for canvas, info in self._canvas_infos.items():
                # someone has switched the canvas on us!  This happens if
                # `savefig` needs to save to a format the previous backend did
                # not support (e.g. saving a figure using an Agg based backend
                # saved to a vector format).
                if canvas is not canvas.figure.canvas:
                    continue
                info["background"] = canvas.copy_from_bbox(canvas.figure.bbox)

    def onmove(self, event):
        axs = [ax for ax in self.axes if ax.contains(event)[0]]
        if self.ignore(event) or not axs or not event.canvas.widgetlock.available(self):
            return
        ax = cbook._topmost_artist(axs)
        xdata, ydata = ((event.xdata, event.ydata) if event.inaxes is ax
                        else ax.transData.inverted().transform((event.x, event.y)))
        for line in self.vlines:
            line.set_xdata((xdata, xdata))
            line.set_visible(self.visible and self.vertOn)
        for line in self.hlines:
            line.set_ydata((ydata, ydata))
            line.set_visible(self.visible and self.horizOn)
        if self.visible and (self.vertOn or self.horizOn):
            self._update()

    def _update(self):
        if self.useblit:
            for canvas, info in self._canvas_infos.items():
                if info["background"]:
                    canvas.restore_region(info["background"])
            if self.vertOn:
                for ax, line in zip(self.axes, self.vlines):
                    ax.draw_artist(line)
            if self.horizOn:
                for ax, line in zip(self.axes, self.hlines):
                    ax.draw_artist(line)
            for canvas in self._canvas_infos:
                canvas.blit()
        else:
            for canvas in self._canvas_infos:
                canvas.draw_idle()


