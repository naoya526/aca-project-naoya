provider "google" {
  project = var.project
  credentials = file(var.gcp_credentials_file)
  region  = var.region
  zone    = var.zone
}

resource "google_compute_instance" "light_nodes" {
  count        = 8
  name         = "light-node-${count.index}"
  machine_type = "e2-medium"
  zone         = var.zone

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-12"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = file("startup-script.sh")

  metadata = {
    ssh-keys = "debian:${file(var.public_key_path)}"
  }

  tags = ["mpi-node"]
}

resource "google_compute_firewall" "mpi-allow-ssh" {
  name    = "allow-ssh-and-mpi"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["22", "8888", "12345"]  # MPI ポートは環境に応じて調整
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["mpi-node"]
}
resource "google_compute_firewall" "allow-all-internal" {
  name    = "allow-all-internal"
  network = "default"

  direction   = "INGRESS"
  source_tags = ["mpi-node"]
  target_tags = ["mpi-node"]

  allow {
    protocol = "all"
  }

  priority = 900
  description = "Allow all internal traffic between MPI nodes"
}