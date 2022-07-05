package com.example.rest.controllers;

import com.example.rest.entities.Results;
import com.example.rest.entities.Vector;
import com.example.rest.loggers.GlobalLogger;
import com.example.rest.services.CalculateService;
import com.example.rest.services.CounterService;
import lombok.AllArgsConstructor;
import org.apache.logging.log4j.Level;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@AllArgsConstructor
@RestController
public class CalculateController {

    private final CalculateService calculateService;
    private final CounterService counterService;

    @GetMapping("/calculate")
    public Results handleCalculationResults(@RequestParam("x1") double x1,
                                            @RequestParam("y1") double y1,
                                            @RequestParam("x2") double x2,
                                            @RequestParam("y2") double y2) {
        var vector = new Vector(x1, y1, x2, y2);
        counterService.increment();
        GlobalLogger.Log(Level.INFO, "CalculateController handleCalculationResults: vector=" + vector);
        return calculateService.calculateResults(vector);
    }

    @PostMapping("/calculate_list")
    public List<Results> handleCalculationResultsList(@RequestBody List<Vector> vectors) {
        counterService.increment();
        GlobalLogger.Log(Level.INFO, "CalculateController handleCalculationResultsList: vectors=" + vectors);
        return calculateService.calculateResultsList(vectors);
    }

    @GetMapping("/counter")
    public long handleCounter() {
        long count = counterService.getCount();
        GlobalLogger.Log(Level.INFO, "CalculateController handleCounter: count=" + count);
        return count;
    }

}
