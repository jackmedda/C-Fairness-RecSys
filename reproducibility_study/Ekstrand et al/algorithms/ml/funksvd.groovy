import org.grouplens.lenskit.iterative.IterationCount
import org.lenskit.api.ItemScorer
import org.lenskit.bias.BiasModel
import org.lenskit.bias.UserItemBiasModel
import org.lenskit.mf.funksvd.FeatureCount
import org.lenskit.mf.funksvd.FunkSVDItemScorer

bind ItemScorer to FunkSVDItemScorer
set FeatureCount to 40
set IterationCount to 150

bind BiasModel to UserItemBiasModel