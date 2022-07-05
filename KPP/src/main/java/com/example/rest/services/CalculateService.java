package com.example.rest.services;

import com.example.rest.cache.Cache;
import com.example.rest.entities.Results;
import com.example.rest.entities.Vector;
import com.example.rest.loggers.GlobalLogger;
import lombok.AllArgsConstructor;
import org.apache.logging.log4j.Level;
import org.jetbrains.annotations.NotNull;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

import static java.util.stream.Collectors.toCollection;

@AllArgsConstructor
@Service
public class CalculateService {

    private Cache cache;

    public double calculateProjectionX(@NotNull Vector vector) {
        return vector.x2() - vector.x1();
    }

    public double calculateProjectionY(@NotNull Vector vector) {
        return vector.y2() - vector.y1();
    }

    public double calculateNormal(@NotNull Vector vector) {
        return Math.sqrt(Math.pow(calculateProjectionX(vector), 2) + Math.pow(calculateProjectionY(vector), 2));
    }

    public Results calculateResults(@NotNull Vector vector) {
        Results results;
        var cachedValue = cache.get(vector);
        if (cachedValue.isPresent()) {
            results = cachedValue.get();
        } else {
            results = new Results(calculateProjectionX(vector), calculateProjectionY(vector), calculateNormal(vector));
            cache.put(vector, results);
        }
        GlobalLogger.Log(Level.INFO, "CalculateService calculateResults: results=" + results);
        return results;
    }

    public List<Results> calculateResultsList(@NotNull List<Vector> list) {
        return list.stream()
                .map(this::calculateResults)
                .collect(toCollection(ArrayList::new));
    }

}
