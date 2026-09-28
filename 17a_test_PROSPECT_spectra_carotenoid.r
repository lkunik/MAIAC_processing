# Install the package (if needed)
# remotes::install_github("jbferet/prospect")

library(prospect)

# Generate reflectance with carotenoids only
# Set chl and ant to 0, keep car non-zero
lrt_car_only <- prospect(
  n_struct = 1.5,
  chl = 0,      # no chlorophyll
  car = 8,      # carotenoids only
  ant = 0,      # no anthocyanins
  ewt = 0.01,
  lma = 0.008
)


# Plot reflectance vs wavelength
plot(lrt_car_only$wvl, lrt_car_only$reflectance,
     type = "l",
     xlab = "Wavelength (nm)",
     ylab = "Reflectance",
     main = "Leaf Reflectance with Carotenoids Only",
     lwd = 2)

# If you want to compare with multiple carotenoid levels
car_values <- c(0, 5, 10, 20)
colors <- c("black", "orange", "darkorange", "red")

plot(NA, xlim = c(400, 2500), ylim = c(0, 0.6),
     xlab = "Wavelength (nm)",
     ylab = "Reflectance",
     main = "Leaf Reflectance vs Carotenoid Content")

for (i in seq_along(car_values)) {
  lrt <- prospect(chl = 0, car = car_values[i], ant = 0, ewt = 0.01, lma = 0.008)
  lines(lrt$wvl, lrt$reflectance, col = colors[i], lwd = 2)
}

legend("topright", 
       legend = paste("car =", car_values),
       col = colors, lwd = 2)