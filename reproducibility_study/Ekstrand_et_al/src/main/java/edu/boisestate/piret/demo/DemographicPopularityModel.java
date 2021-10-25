package edu.boisestate.piret.demo;

import com.google.common.collect.ImmutableList;
import it.unimi.dsi.fastutil.longs.Long2DoubleMap;
import it.unimi.dsi.fastutil.longs.Long2DoubleMaps;
import org.grouplens.grapht.annotation.DefaultProvider;
import org.lenskit.data.entities.Entity;
import org.lenskit.inject.Shareable;

import java.io.Serializable;
import java.util.List;
import java.util.Map;

/**
 * Demographic-stratified popularity model.
 */
@Shareable
@DefaultProvider(DemographicPopularityModelProvider.class)
public class DemographicPopularityModel implements Serializable {
    private final Map<ImmutableList<Object>, Long2DoubleMap> stratifiedPopularityRanks;
    private final List<String> attributes;

    public DemographicPopularityModel(Map<ImmutableList<Object>,Long2DoubleMap> data, List<String> attrs) {
        stratifiedPopularityRanks = data;
        attributes = attrs;
    }

    /**
     * Get the item popularity data for a user's demographic strata.
     * @return The item popularity data for the user's strata.
     */
    public Long2DoubleMap getUserStrataPopularities(Entity user) {
        ImmutableList<Object> strata = extractStrata(user, attributes);
        Long2DoubleMap pop = stratifiedPopularityRanks.get(strata);
        return pop != null ? pop : Long2DoubleMaps.EMPTY_MAP;
    }

    static ImmutableList<Object> extractStrata(Entity user, List<String> attrs) {
        ImmutableList.Builder<Object> lb = ImmutableList.builder();
        for (String a: attrs) {
            lb.add(user.get(a));
        }
        return lb.build();
    }
}
