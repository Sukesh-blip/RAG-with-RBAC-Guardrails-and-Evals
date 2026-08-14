"""
Role-based access control logic.
Maps a requesting user's role to the set of document 'role' tags
they're allowed to retrieve from the vector store.
"""

ROLE_ACCESS = {
    "finance": ["finance", "general", "mixed"],
    "hr": ["hr", "general", "mixed"],
    "c_level": ["finance", "hr", "general", "mixed"],
    "employee": ["general"],
}

VALID_ROLES = set(ROLE_ACCESS.keys())


def get_allowed_doc_roles(user_role: str) -> list[str]:
    """
    Given a user's role, return the list of document-role tags
    they are permitted to retrieve.
    """
    if user_role not in ROLE_ACCESS:
        raise ValueError(
            f"Unknown role '{user_role}'. Valid roles: {sorted(VALID_ROLES)}"
        )
    return ROLE_ACCESS[user_role]