package com.example.rest.cache;

import com.example.rest.entities.Results;
import com.example.rest.entities.Vector;
import com.example.rest.loggers.GlobalLogger;
import org.apache.logging.log4j.Level;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Optional;

@Component
public class Cache {

    private final HashMap<Vector, Results> map = new HashMap<>();

    public void put(Vector vector, Results results) {
        map.put(vector, results);
        GlobalLogger.Log(Level.INFO, "Cache put: key=" + vector + ", value=" + results);
    }

    public Optional<Results> get(Vector vector) {
        Optional<Results> results = Optional.empty();
        if (map.containsKey(vector)) {
            results = Optional.of(map.get(vector));
        }
        GlobalLogger.Log(Level.INFO, "Cache get: key=" + vector + ", value=" + results);
        return results;
    }

}
