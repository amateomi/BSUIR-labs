package com.example.rest;

import com.example.rest.cache.Cache;
import com.example.rest.entities.Results;
import com.example.rest.entities.Vector;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

@SpringBootTest
public class CacheTests {

    @Autowired
    private Cache cache;

    @Test
    void testEmpty() {
        var vector = new Vector(1.0, 2.0, 3.0, 4.0);
        assertTrue(cache.get(vector).isEmpty(), "Cache must be empty");
    }

    @Test
    void testInsertion() {
        var vector = new Vector(1.0, 2.0, 3.0, 4.0);
        var results = new Results(1.0, 2.0, 3.0);
        cache.put(vector, results);

        var cachedResults = cache.get(vector);
        assertTrue(cachedResults.isPresent(), "Cache must contain pair");
        assertEquals(results, cachedResults.get(), "Cached data mustn't be corrupted");
    }

}
