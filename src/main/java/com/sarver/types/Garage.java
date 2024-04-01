package com.sarver.types;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;

public class Garage {
    private boolean isOpened;
    private String name;
    private ArrayList<Person> users;
    private ArrayList<Car> cars;
    private ArrayList<String> plates;
    private int pinID;

    public Garage(String name) {
        this.name = name;
        this.isOpened = false;
        this.users = new ArrayList<>();
        this.cars = new ArrayList<>();
        this.plates = new ArrayList<>();
        this.pinID = name.hashCode();
    }

    public Garage(String name, Person user) {
        this.name = name;
        this.isOpened = false;
        this.users = new ArrayList<>();
        this.users.add(user);
        this.cars = new ArrayList<>();
        this.plates = new ArrayList<>();
        this.pinID = name.hashCode();
    }

    public Garage(String name, Person user, Car car) {
        this.name = name;
        this.isOpened = false;
        this.users = new ArrayList<>();
        this.users.add(user);
        this.cars = new ArrayList<>();
        this.cars.add(car);
        this.plates = new ArrayList<>();
        this.plates.add(car.getPlate());
        this.pinID = name.hashCode();
    }

    public Garage(String name, Person user, ArrayList<Car> cars) {
        this.name = name;
        this.isOpened = false;
        this.users = new ArrayList<>();
        this.users.add(user);
        this.cars = cars;
        this.plates = new ArrayList<>();
        for (Car car : cars) {
            this.plates.add(car.getPlate());
        }
        this.pinID = name.hashCode();
    }

    public boolean isOpened() {
        return isOpened;
    }

    public String getName() {
        return name;
    }

    public ArrayList<Person> getUsers() {
        return users;
    }

    public ArrayList<Car> getCars() {
        return cars;
    }

    public ArrayList<String> getPlates() {
        return plates;
    }

    public int getPinID() {
        return pinID;
    }

    public void open() {
        this.isOpened = true;
    }

    public void close() {
        this.isOpened = false;
    }

    public void addUser(Person user) {
        this.users.add(user);
    }

    public void removeUser(Person user) {
        this.users.remove(user);
    }

    public void addCar(Car car) {
        this.cars.add(car);
        this.plates.add(car.getPlate());
    }

    public void removeCar(Car car) {
        this.cars.remove(car);
        this.plates.remove(car.getPlate());
    }

    public boolean isApprovedCar(Car car) {
        return this.plates.contains(car.getPlate()) || this.cars.contains(car);
    }

    public boolean isApprovedUser(Person user) {
        return this.users.contains(user);
    }

    public boolean isApprovedPlate(String plate) {
        return this.plates.contains(plate);
    }

    public boolean equals(@NotNull Garage other) {
        return this.name.equals(other.getName()) && this.pinID == other.getPinID();
    }

    public String toString() {
        return this.name + " garage with pin " + this.pinID;
    }
}
