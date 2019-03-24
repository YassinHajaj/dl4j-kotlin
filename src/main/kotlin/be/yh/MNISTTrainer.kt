package be.yh

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam

fun main(args: Array<String>) {
    val uiServer = UIServer.getInstance()
    val statsStorage = InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    val numRows = 28
    val numColumns = 28
    val outputNum = 10
    val batchSize = 128
    val rngSeed = 123
    val numEpochs = 5

    val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)

    println("Build model....")
    val multiLayerConfiguration = NeuralNetConfiguration.Builder()
        .seed(rngSeed.toLong())
        .updater(Adam())
        .list()
        .layer(
            DenseLayer.Builder() //create the first, input layer with xavier initialization
                .nIn(numRows * numColumns)
                .nOut(1000)
                .activation(Activation.RELU)
                .build()
        )
        .layer(
            OutputLayer.Builder() //create hidden layer
                .nOut(outputNum)
                .activation(Activation.SOFTMAX)
                .build()
        )
        .build()

    val model = MultiLayerNetwork(multiLayerConfiguration)
    model.init()
    model.setListeners(StatsListener(statsStorage))

    println("Train model....")
    for (i in 0 until numEpochs) {
        model.fit(mnistTrain)
    }

    println("Evaluate model....")
    val eval = Evaluation(outputNum) //create an evaluation object with 10 possible classes
    while (mnistTest.hasNext()) {
        val next = mnistTest.next()
        val output = model.output(next.features) //get the networks prediction
        eval.eval(next.labels, output) //check the prediction against the true class
    }

    println(eval.stats())
}

