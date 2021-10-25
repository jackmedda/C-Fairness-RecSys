package edu.boisestate.piret.demo;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import it.unimi.dsi.fastutil.longs.Long2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import org.lenskit.data.dao.DataAccessObject;
import org.lenskit.data.entities.CommonTypes;
import org.lenskit.data.entities.Entity;
import org.lenskit.data.ratings.Rating;
import org.lenskit.inject.Transient;
import org.lenskit.util.io.ObjectStream;
import org.lenskit.util.math.Vectors;

import javax.inject.Inject;
import javax.inject.Provider;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Build demographic popularity models.
 */
public class DemographicPopularityModelProvider implements Provider<DemographicPopularityModel> {
    private final DataAccessObject dao;
    private final List<String> attributes;

    @Inject
    public DemographicPopularityModelProvider(@Transient DataAccessObject dao, @Strata String attrs) {
        this.dao = dao;
        attributes = Lists.newArrayList(attrs.split("\\s*,\\s*"));
    }

    @Override
    public DemographicPopularityModel get() {
        Long2ObjectMap<ImmutableList<Object>> userStrata = new Long2ObjectOpenHashMap<>();
        try (ObjectStream<Entity> users = dao.query(CommonTypes.USER).stream()) {
            for (Entity user: users) {
                userStrata.put(user.getId(), DemographicPopularityModel.extractStrata(user, attributes));
            }
        }

        Map<ImmutableList<Object>, Long2DoubleOpenHashMap> counts = new HashMap<>();

        try (ObjectStream<Rating> ratings = dao.query(Rating.class).stream()) {
            for (Rating r: ratings) {
                ImmutableList<Object> strata = userStrata.get(r.getUserId());
                Long2DoubleOpenHashMap sc = counts.get(strata);
                if (sc == null) {
                    sc = new Long2DoubleOpenHashMap();
                    counts.put(strata, sc);
                }
                sc.addTo(r.getItemId(), 1);
            }
        }

        Map<ImmutableList<Object>, Long2DoubleMap> popData =
                Maps.transformValues(counts, (ic) -> {
                    double sum = Vectors.sum(ic);
                    return Vectors.multiplyScalar(ic, 1.0 / sum);
                });
        return new DemographicPopularityModel(popData, attributes);
    }
}
