package edu.boisestate.piret.demo;

import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import org.lenskit.transform.normalize.VectorNormalizer;
import org.lenskit.util.InvertibleFunction;
import org.lenskit.util.math.Vectors;

import javax.annotation.Nullable;

/**
 * Take the log of each element of a vector.
 */
public class LogVectorNormalizer implements VectorNormalizer {
    @Override
    public InvertibleFunction<Long2DoubleMap, Long2DoubleMap> makeTransformation(Long2DoubleMap reference) {
        return new LogTransform();
    }

    private static class LogTransform implements InvertibleFunction<Long2DoubleMap, Long2DoubleMap> {
        @Override
        public Long2DoubleMap unapply(Long2DoubleMap input) {
            return Vectors.transform(input, (n) -> Math.pow(10, n));
        }

        @Nullable
        @Override
        public Long2DoubleMap apply(@Nullable Long2DoubleMap input) {
            return Vectors.transform(input, Math::log10);
        }
    }
}
