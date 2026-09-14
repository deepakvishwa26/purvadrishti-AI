"""
Suspect generator module.

Generates synthetic suspect information associated with each fraud case.
All personal information is fully synthetic and uses safe domains.

Suspect data is operational/linkage information.
Raw PII must NOT directly become ML features.
"""

from faker import Faker


def create_suspect_generator(seed):
    """Create a Faker instance for suspect data generation."""
    fake = Faker("en_IN")
    Faker.seed(seed)
    return fake


def generate_suspect(complaint, fake, rng, suspect_counter):
    """
    Generate a synthetic suspect record for a complaint.

    Args:
        complaint: Complaint dict.
        fake: Faker instance.
        rng: numpy RandomState.
        suspect_counter: Current suspect counter for unique IDs.

    Returns:
        tuple: (suspect_dict, updated suspect_counter)
    """
    suspect_counter += 1
    suspect_id = f"SUS{str(suspect_counter).zfill(8)}"

    # Synthetic mobile: Indian format (10 digits starting with 6-9)
    mobile_prefix = rng.choice(["6", "7", "8", "9"])
    mobile_rest = "".join([str(rng.randint(0, 10)) for _ in range(9)])
    suspect_mobile = mobile_prefix + mobile_rest

    # Synthetic email with safe domain
    username = fake.user_name()
    suspect_email = f"{username}@example.test"

    # Synthetic bank account (16 digit)
    suspect_bank_account = "".join([str(rng.randint(0, 10)) for _ in range(16)])

    # Synthetic address
    suspect_address = fake.address().replace("\n", ", ")

    # Synthetic social handle
    handle_prefix = rng.choice(["@user_", "@syn_", "@fake_"])
    suspect_url_social_handle = f"{handle_prefix}{fake.user_name()}"

    suspect = {
        "suspect_id": suspect_id,
        "complaint_id": complaint["complaint_id"],
        "suspect_mobile": suspect_mobile,
        "suspect_email": suspect_email,
        "suspect_bank_account": suspect_bank_account,
        "suspect_address": suspect_address,
        "suspect_url_social_handle": suspect_url_social_handle,
    }

    return suspect, suspect_counter
