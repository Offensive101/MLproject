/*******************************************************************************
 * Copyright (c) 2011 Christian Trutz and others.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     David Green <david.green@tasktop.com> - initial contribution
 *     Christian Trutz <christian.trutz@gmail.com> - initial contribution
 *******************************************************************************/
package org.eclipse.mylyn.github.ui.internal;

import static junit.framework.Assert.assertEquals;
import static junit.framework.Assert.assertNotNull;

import org.eclipse.jface.text.Region;
import org.eclipse.jface.text.hyperlink.IHyperlink;
import org.eclipse.mylyn.github.internal.GitHub;
import org.eclipse.mylyn.tasks.core.TaskRepository;
import org.junit.Before;
import org.junit.Test;

/**
 * Headless test for {@link GitHubRepositoryConnectorUI}
 * 
 * @author Christian Trutz <christian.trutz@gmail.com>
 */
@SuppressWarnings("restriction")
public class GitHubRepositoryConnectorUIHeadlessTest {

	private GitHubRepositoryConnectorUI connectorUI;

	private TaskRepository repository;

	@Before
	public void before() {
		connectorUI = new GitHubRepositoryConnectorUI();
		repository = new TaskRepository(GitHub.CONNECTOR_KIND,
				GitHub.createGitHubUrl("foo", "bar"));
	}

	@Test
	public void testFindHyperlinksTaskRepositoryStringIntInt() {
		IHyperlink[] hyperlinks = connectorUI.findHyperlinks(repository,
				"one #2 three", -1, 0);
		assertNotNull(hyperlinks);
		assertEquals(1, hyperlinks.length);
		assertEquals(new Region(4, 2), hyperlinks[0].getHyperlinkRegion());

		hyperlinks = connectorUI.findHyperlinks(repository, "one #2 three", -1,
				4);
		assertNotNull(hyperlinks);
		assertEquals(1, hyperlinks.length);
		assertEquals(new Region(8, 2), hyperlinks[0].getHyperlinkRegion());
	}

}
