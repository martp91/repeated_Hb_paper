def move_twiny_xaxis_to_bottom(ax, offset=-0.2):
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")

    # Offset the twin axis below the host
    ax.spines["bottom"].set_position(("axes", offset))

    # Turn on the frame for the twin axis, but then hide all 
    # but the bottom spine
    ax.set_frame_on(True)
    ax.patch.set_visible(False)

    # as @ali14 pointed out, for python3, use this
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.spines["bottom"].set_visible(True)

def add_gdl_axis_below(ax, offset=-0.2):
    def mmol_to_gdl(mmol):
        return mmol/0.621
    def convert_ax_mmol_to_gdl(ax1, ax2):
        """
        Update second axis according with first axis.
        """
        x1, x2 = ax1.get_xlim()
        ax2.set_xlim(mmol_to_gdl(x1), mmol_to_gdl(x2))
        ax2.figure.canvas.draw()

    ax_gdl = ax.twiny()
    ax.callbacks.connect("xlim_changed", lambda ax: convert_ax_mmol_to_gdl(ax, ax_gdl))
    move_twiny_xaxis_to_bottom(ax_gdl, offset)
    ax_gdl.set_xlabel('Hb [g/dL]')
    return ax_gdl