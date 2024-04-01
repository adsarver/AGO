package com.sarver.types;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;

public class Car {
    private String make;
    private String model;
    private String color;
    private ArrayList<Person> drivers;
    private String plate;

    public Car(String make, String model, String color, String plate) {
        this.make = make;
        this.model = model;
        this.color = color;
        this.plate = plate;
        this.drivers = new ArrayList<>();
    }

    public Car(String make, String model, String color, String plate, Person driver) {
        this.make = make;
        this.model = model;
        this.color = color;
        this.plate = plate;
        this.drivers = new ArrayList<>();
        this.drivers.add(driver);
    }

    public void addDriver(Person driver) {
        this.drivers.add(driver);
    }

    public void removeDriver(Person driver) {
        this.drivers.remove(driver);
    }

    public String getMake() {
        return this.make;
    }

    public String getModel() {
        return this.model;
    }

    public String getColor() {
        return this.color;
    }

    public String getPlate() {
        return this.plate;
    }

    public ArrayList<Person> getDrivers() {
        return this.drivers;
    }

    public boolean equals(@NotNull Car other) {
        return this.make.equals(other.getMake()) && this.model.equals(other.getModel()) && this.color.equals(other.getColor()) && this.plate.equals(other.getPlate());
    }

    public String toString() {
        return this.color + " " + this.make + " " + this.model + " with plate " + this.plate;
    }
}
