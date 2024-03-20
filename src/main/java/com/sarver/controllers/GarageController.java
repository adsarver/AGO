package com.sarver.controllers;

import com.sarver.types.Car;
import com.sarver.types.Garage;
import com.sarver.types.Person;
import com.sarver.types.ImageData;

import java.util.ArrayList;

public class GarageController {
    private ArrayList<Garage> garages;
    private Car capturedCar;
    private Person capturedFace;
    private Garage currentGarage;

    public void initialize() {
        this.garages = new ArrayList<Garage>();
    }

    public Garage getImageData() {
        String garageName = "placeholder";
        //gets image data from python, then matches it to a garage;
        return garages.stream().filter(garage -> garage.getName().contains(garageName))
                .findFirst().orElse(null);
    }

    public boolean isApproved(Garage garage) {
        int approvals = 0;
        if(garage.isApprovedCar(capturedCar)) approvals++;
        if(garage.isApprovedUser(capturedFace)) approvals++;
        if(garage.isApprovedPlate(capturedCar.getPlate())) approvals++;
        return approvals >= 2;
    }
}
