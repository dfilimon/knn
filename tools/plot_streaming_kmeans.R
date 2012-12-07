PlotStreamingKMeans <- function(infile) {
    print(infile)

    library(package=ggplot2)

    cur_dev <- dev.cur()
    skm <- read.table(infile, skip=1, fill=TRUE)

    plot_file <- paste(infile, '-numDataPointsSeen.png', sep='')
    print(plot_file)
    png(plot_file)
    p <- qplot(1:length(skm[,1]), skm[,1], xlab='numDataPoints',
               ylab='numDataPointsSeen', geom=c('point'))
    print(p)
    dev.off()

    plot_file <- paste(infile, '-estimatedNumClusters.png', sep='')
    print(plot_file)
    png(plot_file)
    p <- qplot(1:length(skm[,2]), skm[,2], xlab='numDataPoints',
               ylab='estimatedNumClusters', geom=c('point'))
    print(p)
    dev.off()

    plot_file <- paste(infile, '-distanceCutoff.png', sep='')
    print(plot_file)
    png(plot_file)
    p <- qplot(1:length(skm[,3]), skm[,3], xlab='numDataPoints',
               ylab='distanceCutoff', geom=c('point'))
    print(p)
    dev.off()

    plot_file <- paste(infile, '-numCentroids.png', sep='')
    print(plot_file)
    png(plot_file)
    p <- qplot(1:length(skm[,4]), skm[,4], xlab='numDataPoints',
               ylab='numCentroids', geom=c('point'))
    print(p)
    dev.off()

    dev.set(cur_dev)

    return(skm)
}
