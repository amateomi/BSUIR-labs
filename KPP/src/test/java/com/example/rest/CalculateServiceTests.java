package com.example.rest;

import com.example.rest.entities.Results;
import com.example.rest.entities.Vector;
import com.example.rest.services.CalculateService;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.util.ArrayList;
import java.util.Arrays;

import static java.util.stream.Collectors.toCollection;
import static org.junit.jupiter.api.Assertions.assertEquals;

@SpringBootTest
public class CalculateServiceTests {

    @Autowired
    private CalculateService calculateService;

    @Test
    void testCalculateProjectionXPositiveDirection() {
        var vector = new Vector(1.0, 2.0, 3.0, 4.0);
        var results = calculateService.calculateProjectionX(vector);
        assertEquals(2.0, results, "Must be positive value");
    }

    @Test
    void testCalculateProjectionXNegativeDirection() {
        var vector = new Vector(4.0, 3.0, 2.0, 1.0);
        var results = calculateService.calculateProjectionX(vector);
        assertEquals(-2.0, results, "Must be negative value");
    }

    @Test
    void testCalculateProjectionYPositiveDirection() {
        var vector = new Vector(-8.0, -5.0, -2.0, 1.0);
        var results = calculateService.calculateProjectionY(vector);
        assertEquals(6.0, results, "Must be positive value");
    }

    @Test
    void testCalculateProjectionYNegativeDirection() {
        var vector = new Vector(1.0, -2.0, -5.0, -8.0);
        var results = calculateService.calculateProjectionY(vector);
        assertEquals(-6.0, results, "Must be negative value");
    }

    @Test
    void testCalculateNormal() {
        var vector = new Vector(1.0, 2.0, 4.0, 6.0);
        var results = calculateService.calculateNormal(vector);
        assertEquals(5.0, results, "Simple egyptian triangle with cathetus 3 and 4");
    }

    @Test
    void testCalculateResults() {
        var vector = new Vector(-2.0, 2.0, 4.0, 10.0);
        var expected = new Results(6.0, 8.0, 10.0);
        var results = calculateService.calculateResults(vector);
        assertEquals(expected, results);
    }

    @Test
    void testCalculateResultsList() {
        var vectors = new ArrayList<>(
                Arrays.asList(
                        new Vector(1.0, 2.0, 4.0, 6.0),
                        new Vector(1.0, -2.0, -5.0, -8.0),
                        new Vector(-2.0, 2.0, 4.0, 10.0)));
        var expected = vectors.stream()
                .map(vector -> calculateService.calculateResults(vector))
                .collect(toCollection(ArrayList::new));
        var results = calculateService.calculateResultsList(vectors);
        assertEquals(expected, results);
    }

}
