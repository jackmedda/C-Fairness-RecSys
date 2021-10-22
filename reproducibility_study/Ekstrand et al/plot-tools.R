make.plot = function(name, plot_expr, type="pdf", width, height, ...) {
    dir.create("build/figures", recursive = TRUE, showWarnings = FALSE)
    if (identical(type, "pdf")) {
        pdf(sprintf("build/figures/%s.pdf", name),
            width=width, height=height, ...)
    } else if (identical(type, "png")) {
        png(sprintf("build/figures/%s.png", name),
            width=width, height=height, units="in", res=300, ...)
    } else {
        stop("invalid plot type")
    }
    print(plot_expr)
    dev.off()
    plot_expr
}

theme_paper = function() {
    theme_minimal() + theme(panel.border=element_rect(linetype="solid", color="grey", fill=NA))
}
